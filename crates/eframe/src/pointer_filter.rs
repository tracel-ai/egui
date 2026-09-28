//! Ueye patch (DESIGN.md 9.4): keeping the input of pointer moves that run
//! no UI pass, shared by the native and web runners.

/// Ueye patch: before a filtered pointer move is queued, drops the queued
/// moves it replaces, so moves that run no pass do not pile up until the
/// next one. Only the trailing run of moves is touched: order around other
/// events is kept.
pub(crate) fn drop_superseded_pointer_move(input: &mut egui::RawInput) {
    let run = input
        .events
        .iter()
        .rev()
        .take_while(|event| {
            matches!(
                event,
                egui::Event::PointerMoved(_) | egui::Event::MouseMoved(_)
            )
        })
        .count();
    let start = input.events.len() - run;
    let mut index = start;
    input.events.retain(|event| {
        let keep = index < start || !matches!(event, egui::Event::PointerMoved(_));
        index += 1;
        keep
    });
}

/// Ueye patch: whether queued input holds nothing but pointer moves. Moves
/// that need a UI pass mark their window for one (`needs_full_ui`), so a
/// paint-only frame can replay over moves that were filtered out.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn only_pointer_moves(events: &[egui::Event]) -> bool {
    events.iter().all(|event| {
        matches!(
            event,
            egui::Event::PointerMoved(_) | egui::Event::MouseMoved(_) | egui::Event::PointerGone
        )
    })
}

/// Ueye patch: after a raw mouse delta that runs no pass is queued, merges
/// it into the previous one of the trailing run of moves, for the same
/// reason.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn merge_mouse_motion(input: &mut egui::RawInput) {
    let Some(egui::Event::MouseMoved(delta)) = input.events.last().cloned() else {
        return;
    };
    let len = input.events.len();
    let previous = input.events[..len - 1]
        .iter()
        .rev()
        .take_while(|event| {
            matches!(
                event,
                egui::Event::PointerMoved(_) | egui::Event::MouseMoved(_)
            )
        })
        .position(|event| matches!(event, egui::Event::MouseMoved(_)));
    if let Some(offset) = previous
        && let egui::Event::MouseMoved(sum) = &mut input.events[len - 2 - offset]
    {
        *sum += delta;
        input.events.pop();
    }
}
