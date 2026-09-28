use pyo3::exceptions::{PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::time::Instant;

type Edge = (usize, usize);
type PyEvent = (String, usize, usize, f64);

#[derive(Clone, Debug)]
struct PendingTerm {
    u: usize,
    v: usize,
    angle: f64,
}

#[derive(Clone, Debug)]
enum Event {
    Rzz(usize, usize, f64),
    Swap(usize, usize),
}

#[derive(Clone, Debug)]
struct State {
    where_: Vec<usize>,
    pending: Vec<PendingTerm>,
    events: Vec<Event>,
    swaps: usize,
    steps: usize,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
struct Signature {
    where_: Vec<usize>,
    pending_pairs: Vec<Edge>,
}

#[derive(Clone, Copy, Debug)]
struct Score(f64, f64, f64);

#[derive(Debug)]
struct ChipData {
    chip_size: usize,
    edges: Vec<Edge>,
    neighbors: Vec<Vec<usize>>,
    distances: Vec<Vec<usize>>,
    paths: Vec<Vec<Vec<Edge>>>,
    degree: Vec<usize>,
}

#[derive(Debug)]
struct RouteOutput {
    events: Vec<Event>,
    where_: Vec<usize>,
    swaps: usize,
}

#[derive(Debug)]
enum CoreError {
    BudgetExceeded,
    RoutingFailed,
    InvalidInput(String),
}

fn canonical_edge(a: usize, b: usize) -> Edge {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn cmp_f64(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

fn cmp_score(a: Score, b: Score) -> Ordering {
    cmp_f64(a.0, b.0)
        .then_with(|| cmp_f64(a.1, b.1))
        .then_with(|| cmp_f64(a.2, b.2))
}

fn budget_exceeded(start: Instant, max_seconds: Option<f64>) -> bool {
    match max_seconds {
        None => false,
        Some(seconds) => start.elapsed().as_secs_f64() >= seconds,
    }
}

fn build_chip_data(chip_size: usize, raw_edges: &[Edge]) -> Result<ChipData, CoreError> {
    if chip_size == 0 {
        return Err(CoreError::InvalidInput(
            "chip_size must be positive".to_string(),
        ));
    }

    let mut edge_set = BTreeSet::new();
    let mut neighbors = vec![Vec::<usize>::new(); chip_size];

    for &(a, b) in raw_edges {
        if a >= chip_size || b >= chip_size {
            return Err(CoreError::InvalidInput(format!(
                "coupling edge ({a}, {b}) is outside chip_size={chip_size}"
            )));
        }
        if a == b {
            continue;
        }
        let edge = canonical_edge(a, b);
        if edge_set.insert(edge) {
            neighbors[a].push(b);
            neighbors[b].push(a);
        }
    }

    for values in &mut neighbors {
        values.sort_unstable();
        values.dedup();
    }

    let edges: Vec<Edge> = edge_set.into_iter().collect();
    let degree: Vec<usize> = neighbors.iter().map(Vec::len).collect();
    let mut distances = vec![vec![usize::MAX; chip_size]; chip_size];
    let mut paths = vec![vec![Vec::<Edge>::new(); chip_size]; chip_size];

    for start in 0..chip_size {
        let mut previous = vec![None::<usize>; chip_size];
        let mut queue = VecDeque::new();
        previous[start] = Some(start);
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            for &next in &neighbors[current] {
                if previous[next].is_none() {
                    previous[next] = Some(current);
                    queue.push_back(next);
                }
            }
        }

        for finish in 0..chip_size {
            if previous[finish].is_none() {
                return Err(CoreError::InvalidInput(
                    "The hardware coupling map must be connected.".to_string(),
                ));
            }

            let mut path = Vec::new();
            let mut current = finish;
            while current != start {
                let parent = previous[current].expect("connected graph has predecessor");
                path.push(canonical_edge(parent, current));
                current = parent;
            }
            path.reverse();
            distances[start][finish] = path.len();
            paths[start][finish] = path;
        }
    }

    Ok(ChipData {
        chip_size,
        edges,
        neighbors,
        distances,
        paths,
        degree,
    })
}

fn normalise(mut state: State, chip: &ChipData) -> State {
    let mut remaining = Vec::with_capacity(state.pending.len());

    // state.pending is kept in ascending (u, v) order, matching V1's sorted(dict).
    for term in state.pending.into_iter() {
        let a = state.where_[term.u];
        let b = state.where_[term.v];
        if chip.distances[a][b] == 1 {
            state.events.push(Event::Rzz(a, b, term.angle));
        } else {
            remaining.push(term);
        }
    }

    state.pending = remaining;
    state
}

fn apply_batch(state: &State, batch: &[Edge], at: &[Option<usize>]) -> State {
    let mut candidate = state.clone();
    let mut occupied = at.to_vec();

    for &(x, y) in batch {
        let left = occupied[x];
        let right = occupied[y];
        occupied[x] = right;
        occupied[y] = left;

        if let Some(logical) = left {
            candidate.where_[logical] = y;
        }
        if let Some(logical) = right {
            candidate.where_[logical] = x;
        }
        candidate.events.push(Event::Swap(x, y));
    }

    candidate.swaps += batch.len();
    candidate.steps += 1;
    candidate
}

fn incident_count(term: &PendingTerm, pending: &[PendingTerm]) -> usize {
    pending
        .iter()
        .filter(|other| {
            term.u == other.u
                || term.u == other.v
                || term.v == other.u
                || term.v == other.v
        })
        .count()
}

fn edge_gain(edge: Edge, state: &State, at: &[Option<usize>], chip: &ChipData) -> isize {
    let (x, y) = edge;
    let mut gain = 0isize;

    for term in &state.pending {
        let before = chip.distances[state.where_[term.u]][state.where_[term.v]] as isize;
        let after_u = if at[x] == Some(term.u) {
            y
        } else if at[y] == Some(term.u) {
            x
        } else {
            state.where_[term.u]
        };
        let after_v = if at[x] == Some(term.v) {
            y
        } else if at[y] == Some(term.v) {
            x
        } else {
            state.where_[term.v]
        };
        gain += before - chip.distances[after_u][after_v] as isize;
    }

    gain
}

fn focus_edges(state: &State, at: &[Option<usize>], chip: &ChipData) -> Vec<Edge> {
    let pending = &state.pending;
    let mut urgency: Vec<(usize, (usize, usize))> = pending
        .iter()
        .enumerate()
        .map(|(index, term)| {
            (
                index,
                (
                    chip.distances[state.where_[term.u]][state.where_[term.v]],
                    incident_count(term, pending),
                ),
            )
        })
        .collect();

    // Python's sorted(key=...) computes each key once and is stable. Do the same.
    // reverse=True means larger (distance, incident_count) keys come first.
    urgency.sort_by(|left, right| right.1.cmp(&left.1));

    let mut candidates = BTreeSet::<Edge>::new();
    for &(index, _) in urgency.iter().take(6) {
        let term = &pending[index];
        let a = state.where_[term.u];
        let b = state.where_[term.v];
        let old = chip.distances[a][b];

        for &(x, y) in &chip.edges {
            if x != a && x != b && y != a && y != b {
                continue;
            }

            let mut moved = if x == a {
                y
            } else if y == a {
                x
            } else {
                a
            };
            let mut other = b;
            if x == b {
                moved = y;
                other = a;
            } else if y == b {
                moved = x;
                other = a;
            }

            if chip.distances[moved][other] < old {
                candidates.insert((x, y));
            }
        }
    }

    if candidates.is_empty() {
        candidates.extend(chip.edges.iter().copied());
    }

    let mut ranked: Vec<(Edge, isize)> = candidates
        .into_iter()
        .map(|edge| (edge, edge_gain(edge, state, at, chip)))
        .collect();
    // Python's sorted(key=...) computes edge_gain once per candidate.
    ranked.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| right.0.cmp(&left.0))
    });
    ranked.into_iter().map(|(edge, _)| edge).collect()
}

fn non_overlapping(batch: &[Edge]) -> bool {
    for i in 0..batch.len() {
        for j in (i + 1)..batch.len() {
            let (a, b) = batch[i];
            let (c, d) = batch[j];
            if a == c || a == d || b == c || b == d {
                return false;
            }
        }
    }
    true
}

fn batches(state: &State, at: &[Option<usize>], chip: &ChipData, max_candidates: usize) -> Vec<Vec<Edge>> {
    let singles_all = focus_edges(state, at, chip);
    let singles: Vec<Edge> = singles_all.into_iter().take(10).collect();
    let mut out = Vec::<Vec<Edge>>::new();

    for &edge in &singles {
        out.push(vec![edge]);
        if out.len() >= max_candidates {
            return out;
        }
    }

    let limit = singles.len().min(8);
    for i in 0..limit {
        for j in (i + 1)..limit {
            let mut batch = vec![singles[i], singles[j]];
            if non_overlapping(&batch) {
                batch.sort_unstable();
                out.push(batch);
                if out.len() >= max_candidates {
                    return out;
                }
            }
        }
    }

    for i in 0..limit {
        for j in (i + 1)..limit {
            for k in (j + 1)..limit {
                let mut batch = vec![singles[i], singles[j], singles[k]];
                if non_overlapping(&batch) {
                    batch.sort_unstable();
                    out.push(batch);
                    if out.len() >= max_candidates {
                        return out;
                    }
                }
            }
        }
    }

    out
}

fn score(state: &State, chip: &ChipData) -> Score {
    if state.pending.is_empty() {
        // V1 intentionally returns (swaps, steps, 0.0) for a finished state.
        return Score(state.swaps as f64, state.steps as f64, 0.0);
    }

    let mut remaining = vec![0usize; state.where_.len()];
    for term in &state.pending {
        remaining[term.u] += 1;
        remaining[term.v] += 1;
    }

    let mut edge_load = HashMap::<Edge, usize>::new();
    let mut distance_cost = 0.0f64;
    let mut max_distance = 0usize;

    for term in &state.pending {
        let a = state.where_[term.u];
        let b = state.where_[term.v];
        let distance = chip.distances[a][b];
        let extra = distance.saturating_sub(1) as f64;
        distance_cost += extra * (1.0 + 0.03 * (remaining[term.u] + remaining[term.v]) as f64);
        max_distance = max_distance.max(distance);

        for &edge in &chip.paths[a][b] {
            *edge_load.entry(edge).or_insert(0) += 1;
        }
    }

    let max_load = edge_load.values().copied().max().unwrap_or(0) as f64;
    let squared_sum: usize = edge_load.values().map(|value| value * value).sum();
    let congestion = max_load + 0.15 * squared_sum as f64;

    let mut trap_penalty = 0.0f64;
    for (logical, &count) in remaining.iter().enumerate() {
        let site = state.where_[logical];
        let denominator = chip.degree[site].max(1) as f64;
        trap_penalty += count as f64 / denominator;
    }

    let value = 4.0 * distance_cost
        + 1.5 * max_distance as f64
        + 0.8 * congestion
        + 0.4 * trap_penalty
        + 0.25 * state.swaps as f64
        + 0.15 * state.steps as f64;

    Score(value, state.swaps as f64, state.steps as f64)
}

fn signature(state: &State) -> Signature {
    Signature {
        where_: state.where_.clone(),
        pending_pairs: state.pending.iter().map(|term| (term.u, term.v)).collect(),
    }
}

fn route_layer_core(
    pending_input: Vec<(usize, usize, f64)>,
    where_: Vec<usize>,
    chip_size: usize,
    raw_edges: Vec<Edge>,
    beam_width: usize,
    max_seconds: Option<f64>,
) -> Result<RouteOutput, CoreError> {
    // Start the budget before topology/preprocessing work. V1 starts its global
    // deadline before entering _route_layer, so chip-data construction counts too.
    let start = Instant::now();

    if where_.is_empty() {
        return Err(CoreError::InvalidInput(
            "where must contain at least one logical qubit".to_string(),
        ));
    }
    if where_.iter().any(|&site| site >= chip_size) {
        return Err(CoreError::InvalidInput(
            "where contains a physical site outside chip_size".to_string(),
        ));
    }
    let mut used_sites = BTreeSet::new();
    if where_.iter().any(|&site| !used_sites.insert(site)) {
        return Err(CoreError::InvalidInput(
            "where contains duplicate physical sites".to_string(),
        ));
    }

    let chip = build_chip_data(chip_size, &raw_edges)?;
    debug_assert_eq!(chip.neighbors.len(), chip_size);

    let mut pending = Vec::<PendingTerm>::with_capacity(pending_input.len());
    for (u, v, angle) in pending_input {
        if u >= where_.len() || v >= where_.len() {
            return Err(CoreError::InvalidInput(format!(
                "pending pair ({u}, {v}) is outside the logical layout"
            )));
        }
        pending.push(PendingTerm { u, v, angle });
    }
    pending.sort_by_key(|term| (term.u, term.v));

    let initial_pending_len = pending.len();
    let initial = normalise(
        State {
            where_,
            pending,
            events: Vec::new(),
            swaps: 0,
            steps: 0,
        },
        &chip,
    );

    let mut beam = vec![initial];
    let max_steps = 8usize.max(initial_pending_len * (chip.chip_size + 1));

    for _ in 0..max_steps {
        if budget_exceeded(start, max_seconds) {
            return Err(CoreError::BudgetExceeded);
        }

        // Python dict preserves first insertion order even when an existing value is
        // replaced. The Vec + signature->index map below intentionally reproduces that.
        let mut next_states = Vec::<State>::new();
        let mut next_index = HashMap::<Signature, usize>::new();
        let mut finished = Vec::<State>::new();

        for state in &beam {
            if budget_exceeded(start, max_seconds) {
                return Err(CoreError::BudgetExceeded);
            }
            if state.pending.is_empty() {
                finished.push(state.clone());
                continue;
            }

            let mut at = vec![None::<usize>; chip.chip_size];
            for (logical, &site) in state.where_.iter().enumerate() {
                at[site] = Some(logical);
            }

            for batch in batches(state, &at, &chip, 24) {
                let candidate = normalise(apply_batch(state, &batch, &at), &chip);
                let key = signature(&candidate);

                if let Some(&index) = next_index.get(&key) {
                    if cmp_score(score(&candidate, &chip), score(&next_states[index], &chip))
                        == Ordering::Less
                    {
                        next_states[index] = candidate;
                    }
                } else {
                    let index = next_states.len();
                    next_states.push(candidate);
                    next_index.insert(key, index);
                }
            }
        }

        if !finished.is_empty() {
            let mut best_index = 0usize;
            for index in 1..finished.len() {
                let best_key = (finished[best_index].swaps, finished[best_index].steps);
                let candidate_key = (finished[index].swaps, finished[index].steps);
                if candidate_key < best_key {
                    best_index = index;
                }
            }
            let best = finished.swap_remove(best_index);
            return Ok(RouteOutput {
                events: best.events,
                where_: best.where_,
                swaps: best.swaps,
            });
        }

        if next_states.is_empty() {
            break;
        }

        // Python sorted(key=_score) evaluates the score once per state and is stable.
        let mut scored_states: Vec<(Score, State)> = next_states
            .into_iter()
            .map(|state| (score(&state, &chip), state))
            .collect();
        scored_states.sort_by(|left, right| cmp_score(left.0, right.0));
        beam = scored_states
            .into_iter()
            .take(beam_width)
            .map(|(_, state)| state)
            .collect();
    }

    Err(CoreError::RoutingFailed)
}

fn route_line_layer_core(
    pending_input: Vec<(usize, usize, f64)>,
    where_: Vec<usize>,
    chip_size: usize,
    raw_edges: Vec<Edge>,
    path: Vec<usize>,
) -> Result<RouteOutput, CoreError> {
    let n = where_.len();
    if n < 2 || path.len() != n {
        return Err(CoreError::InvalidInput(
            "The physical line must contain every logical qubit exactly once.".to_string(),
        ));
    }
    let chip = build_chip_data(chip_size, &raw_edges)?;
    let mut path_sites = BTreeSet::new();
    if path.iter().any(|&site| site >= chip_size || !path_sites.insert(site))
        || where_.iter().any(|site| !path_sites.contains(site))
        || where_.iter().copied().collect::<BTreeSet<_>>().len() != n
        || path.windows(2).any(|pair| chip.distances[pair[0]][pair[1]] != 1)
    {
        return Err(CoreError::InvalidInput(
            "The starting layout must occupy a valid, connected physical line.".to_string(),
        ));
    }
    let mut pending = Vec::with_capacity(pending_input.len());
    for (u, v, angle) in pending_input {
        if u >= n || v >= n || u == v || !angle.is_finite() {
            return Err(CoreError::InvalidInput("Invalid ZZ interaction.".to_string()));
        }
        pending.push(PendingTerm { u, v, angle });
    }
    pending.sort_by_key(|term| (term.u, term.v));
    let mut state = normalise(
        State { where_, pending, events: Vec::new(), swaps: 0, steps: 0 },
        &chip,
    );
    // Alternating disjoint SWAP layers make every pair adjacent within n-2 rounds.
    // Stop as soon as all requested ZZ interactions have been executed.
    for round in 0..n.saturating_sub(2) {
        if state.pending.is_empty() {
            break;
        }
        let batch: Vec<Edge> = (round % 2..n - 1)
            .step_by(2)
            .map(|index| canonical_edge(path[index], path[index + 1]))
            .collect();
        let mut at = vec![None; chip_size];
        for (logical, &site) in state.where_.iter().enumerate() {
            at[site] = Some(logical);
        }
        state = normalise(apply_batch(&state, &batch, &at), &chip);
    }
    if !state.pending.is_empty() {
        return Err(CoreError::RoutingFailed);
    }
    Ok(RouteOutput {
        events: state.events,
        where_: state.where_,
        swaps: state.swaps,
    })
}

fn line_meeting_layers(n: usize) -> Vec<Vec<usize>> {
    let mut at: Vec<usize> = (0..n).collect();
    let mut first = vec![vec![usize::MAX; n]; n];
    for round in 0..=n - 2 {
        for pair in at.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if first[a][b] == usize::MAX {
                first[a][b] = round;
                first[b][a] = round;
            }
        }
        if round < n - 2 {
            for index in (round % 2..n - 1).step_by(2) {
                at.swap(index, index + 1);
            }
        }
    }
    first
}

fn next_random(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn line_layout_score(positions: &[usize], edges: &[Edge], meeting: &[Vec<usize>]) -> u64 {
    edges.iter().map(|&(u, v)| 1u64 << meeting[positions[u]][positions[v]]).sum()
}

fn optimise_line_layout_core(
    n: usize, edges: Vec<Edge>, seed: u64, max_seconds: f64,
) -> Result<Vec<usize>, CoreError> {
    if !(2..=30).contains(&n) || !max_seconds.is_finite() || max_seconds <= 0.0
        || edges.iter().any(|&(u, v)| u >= n || v >= n || u == v)
    {
        return Err(CoreError::InvalidInput(
            "Expected 2-30 logical qubits, valid pairs, and a positive time limit.".to_string(),
        ));
    }
    let meeting = line_meeting_layers(n);
    let start = Instant::now();
    let mut rng = seed.max(1);
    let mut best: Vec<usize> = (0..n).collect();
    let mut best_score = line_layout_score(&best, &edges, &meeting);
    let mut positions = best.clone();
    let mut current_score = best_score;
    let mut iteration = 0usize;
    while iteration < 1_000_000 {
        if iteration % 256 == 0 && budget_exceeded(start, Some(max_seconds)) {
            break;
        }
        if iteration > 0 && iteration % 10_000 == 0 {
            positions.clone_from(&best);
            for _ in 0..(n / 3).max(2) {
                let u = (next_random(&mut rng) as usize) % n;
                let v = (next_random(&mut rng) as usize) % n;
                positions.swap(u, v);
            }
            current_score = line_layout_score(&positions, &edges, &meeting);
        }
        let u = (next_random(&mut rng) as usize) % n;
        let v = (next_random(&mut rng) as usize) % n;
        if u != v {
            positions.swap(u, v);
            let trial = line_layout_score(&positions, &edges, &meeting);
            let progress = (iteration % 10_000) as f64 / 10_000.0;
            let temperature = (1u64 << (n - 2).min(22)) as f64 * (1.0 - progress).powi(2) + 1.0;
            let accept = trial <= current_score
                || ((current_score as f64 - trial as f64) / temperature).exp()
                    > (next_random(&mut rng) as f64 / u64::MAX as f64);
            if accept {
                current_score = trial;
                if trial < best_score {
                    best_score = trial;
                    best.clone_from(&positions);
                }
            } else {
                positions.swap(u, v);
            }
        }
        iteration += 1;
    }
    Ok(best)
}

fn line_layout_score_wide(positions: &[usize], edges: &[Edge], meeting: &[Vec<usize>]) -> u128 {
    // Rounds above 30 need more than a 64-bit weighted sum.
    edges.iter().map(|&(u, v)| 1u128 << meeting[positions[u]][positions[v]]).sum()
}

fn optimise_line_layout_wide_core(
    n: usize, edges: Vec<Edge>, seed: u64, max_seconds: f64,
) -> Result<Vec<usize>, CoreError> {
    if !(31..=64).contains(&n) || !max_seconds.is_finite() || max_seconds <= 0.0
        || edges.iter().any(|&(u, v)| u >= n || v >= n || u == v)
    {
        return Err(CoreError::InvalidInput(
            "Expected 31-64 logical qubits, valid pairs, and a positive time limit.".to_string(),
        ));
    }
    let meeting = line_meeting_layers(n);
    let start = Instant::now();
    let mut rng = seed.max(1);
    let mut best: Vec<usize> = (0..n).collect();
    let mut best_score = line_layout_score_wide(&best, &edges, &meeting);
    let mut positions = best.clone();
    let mut current_score = best_score;
    let mut iteration = 0usize;
    while iteration < 1_000_000 {
        if iteration % 256 == 0 && budget_exceeded(start, Some(max_seconds)) {
            break;
        }
        if iteration > 0 && iteration % 10_000 == 0 {
            positions.clone_from(&best);
            for _ in 0..(n / 3).max(2) {
                let u = (next_random(&mut rng) as usize) % n;
                let v = (next_random(&mut rng) as usize) % n;
                positions.swap(u, v);
            }
            current_score = line_layout_score_wide(&positions, &edges, &meeting);
        }
        let u = (next_random(&mut rng) as usize) % n;
        let v = (next_random(&mut rng) as usize) % n;
        if u != v {
            positions.swap(u, v);
            let trial = line_layout_score_wide(&positions, &edges, &meeting);
            let progress = (iteration % 10_000) as f64 / 10_000.0;
            let temperature = (1u64 << (n - 2).min(22)) as f64 * (1.0 - progress).powi(2) + 1.0;
            let accept = trial <= current_score
                || ((current_score as f64 - trial as f64) / temperature).exp()
                    > (next_random(&mut rng) as f64 / u64::MAX as f64);
            if accept {
                current_score = trial;
                if trial < best_score {
                    best_score = trial;
                    best.clone_from(&positions);
                }
            } else {
                positions.swap(u, v);
            }
        }
        iteration += 1;
    }
    Ok(best)
}

#[pyfunction]
#[pyo3(signature = (n, edges, seed=11, max_seconds=0.15))]
fn optimise_line_layout(
    py: Python<'_>, n: usize, edges: Vec<Edge>, seed: u64, max_seconds: f64,
) -> PyResult<Vec<usize>> {
    match py.detach(move || {
        if n <= 30 {
            optimise_line_layout_core(n, edges, seed, max_seconds)
        } else {
            optimise_line_layout_wide_core(n, edges, seed, max_seconds)
        }
    }) {
        Ok(positions) => Ok(positions),
        Err(CoreError::InvalidInput(message)) => Err(PyValueError::new_err(message)),
        Err(_) => unreachable!(),
    }
}

#[pyfunction]
fn route_line_layer(
    py: Python<'_>,
    pending: Vec<(usize, usize, f64)>,
    where_: Vec<usize>,
    chip_size: usize,
    edges: Vec<Edge>,
    path: Vec<usize>,
) -> PyResult<(Vec<PyEvent>, Vec<usize>, usize)> {
    let result = py.detach(move || route_line_layer_core(pending, where_, chip_size, edges, path));
    match result {
        Ok(output) => {
            let events = output.events.into_iter().map(|event| match event {
                Event::Rzz(a, b, angle) => ("rzz".to_string(), a, b, angle),
                Event::Swap(a, b) => ("swap".to_string(), a, b, 0.0),
            }).collect();
            Ok((events, output.where_, output.swaps))
        }
        Err(CoreError::InvalidInput(message)) => Err(PyValueError::new_err(message)),
        Err(CoreError::RoutingFailed) => Err(PyRuntimeError::new_err(
            "The line schedule did not cover every ZZ interaction.",
        )),
        Err(CoreError::BudgetExceeded) => unreachable!(),
    }
}

#[pyfunction]
#[pyo3(signature = (pending, where_, chip_size, edges, beam_width=16, max_seconds=None))]
fn route_layer(
    py: Python<'_>,
    pending: Vec<(usize, usize, f64)>,
    where_: Vec<usize>,
    chip_size: usize,
    edges: Vec<Edge>,
    beam_width: usize,
    max_seconds: Option<f64>,
) -> PyResult<(Vec<PyEvent>, Vec<usize>, usize)> {
    let result = py.detach(move || {
        route_layer_core(
            pending,
            where_,
            chip_size,
            edges,
            beam_width,
            max_seconds,
        )
    });

    match result {
        Ok(output) => {
            let events = output
                .events
                .into_iter()
                .map(|event| match event {
                    Event::Rzz(a, b, angle) => ("rzz".to_string(), a, b, angle),
                    Event::Swap(a, b) => ("swap".to_string(), a, b, 0.0),
                })
                .collect();
            Ok((events, output.where_, output.swaps))
        }
        Err(CoreError::BudgetExceeded) => Err(PyTimeoutError::new_err(
            "Hybrid Rust search used up its time budget.",
        )),
        Err(CoreError::RoutingFailed) => Err(PyRuntimeError::new_err(
            "Hybrid Rust search could not route the whole ZZ layer.",
        )),
        Err(CoreError::InvalidInput(message)) => Err(PyValueError::new_err(message)),
    }
}

#[pymodule]
fn qaoa_v2_rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(route_layer, module)?)?;
    module.add_function(wrap_pyfunction!(route_line_layer, module)?)?;
    module.add_function(wrap_pyfunction!(optimise_line_layout, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
