//! UI Automation, Accessibility Tree, Window Management, and Multi-Monitor Coordinate Transforms.
//!
//! Provides:
//! - Structured UI hierarchy, control identification, supported actions, and focus tracking.
//! - Accurate DPI scaling and coordinate transforms across multi-monitor virtual desktop layouts.
//! - Window movement, resize detection, and pointer location tracking.

use serde::{Deserialize, Serialize};

/// 2D rectangle in logical or physical pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn contains_point(&self, px: i32, py: i32) -> bool {
        px >= self.x
            && px < self.x + self.width as i32
            && py >= self.y
            && py < self.y + self.height as i32
    }

    pub fn to_physical(&self, scale: f32) -> Self {
        Self {
            x: ((self.x as f32) * scale).round() as i32,
            y: ((self.y as f32) * scale).round() as i32,
            width: ((self.width as f32) * scale).round().max(1.0) as u32,
            height: ((self.height as f32) * scale).round().max(1.0) as u32,
        }
    }

    pub fn to_logical(&self, scale: f32) -> Self {
        let safe_scale = if scale <= 0.0 { 1.0 } else { scale };
        Self {
            x: ((self.x as f32) / safe_scale).round() as i32,
            y: ((self.y as f32) / safe_scale).round() as i32,
            width: ((self.width as f32) / safe_scale).round().max(1.0) as u32,
            height: ((self.height as f32) / safe_scale).round().max(1.0) as u32,
        }
    }

    pub fn intersects(&self, other: &Rect) -> bool {
        self.x < other.x + other.width as i32
            && self.x + self.width as i32 > other.x
            && self.y < other.y + other.height as i32
            && self.y + self.height as i32 > other.y
    }
}

/// Description of a display monitor in the virtual desktop space.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MonitorInfo {
    pub monitor_id: String,
    pub name: String,
    pub bounds: Rect,
    pub scale_factor: f32,
    pub is_primary: bool,
}

/// Metadata and bounds for an application window.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WindowInfo {
    pub window_id: u64,
    pub title: String,
    pub process_id: u32,
    pub process_name: String,
    pub bounds: Rect,
    pub monitor_id: String,
    pub display_scale: f32,
    pub is_active: bool,
    pub is_minimized: bool,
}

/// UI Automation node representing a control, text block, or container.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UiElement {
    pub element_id: String,
    pub name: String,
    pub role: String,
    pub text_value: Option<String>,
    pub bounds: Rect,
    pub supported_actions: Vec<String>,
    pub is_focused: bool,
    pub is_enabled: bool,
    pub children: Vec<UiElement>,
}

impl UiElement {
    /// Recursively searches for an element containing the given window-relative point.
    pub fn find_element_at(&self, x: i32, y: i32) -> Option<&UiElement> {
        if !self.bounds.contains_point(x, y) {
            return None;
        }
        for child in &self.children {
            if let Some(found) = child.find_element_at(x, y) {
                return Some(found);
            }
        }
        Some(self)
    }

    /// Recursively collects all text present in this UI subtree.
    pub fn collect_all_text(&self) -> Vec<String> {
        let mut text = Vec::new();
        if let Some(val) = &self.text_value {
            if !val.trim().is_empty() {
                text.push(val.clone());
            }
        } else if !self.name.trim().is_empty() && self.role != "Window" && self.role != "Pane" {
            text.push(self.name.clone());
        }
        for child in &self.children {
            text.extend(child.collect_all_text());
        }
        text
    }

    /// Recursively searches for an element by element_id.
    pub fn find_by_id(&self, target_id: &str) -> Option<&UiElement> {
        if self.element_id == target_id {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find_by_id(target_id) {
                return Some(found);
            }
        }
        None
    }
}

/// Handles coordinate transformations between virtual desktop, window-local, and DPI-scaled pixel spaces.
pub struct CoordinateTransformer;

impl CoordinateTransformer {
    /// Maps a virtual desktop point to window-local coordinates.
    pub fn desktop_to_window(desktop_x: i32, desktop_y: i32, window: &WindowInfo) -> (i32, i32) {
        let local_x = desktop_x - window.bounds.x;
        let local_y = desktop_y - window.bounds.y;
        (local_x, local_y)
    }

    /// Maps a window-local coordinate to virtual desktop coordinates.
    pub fn window_to_desktop(local_x: i32, local_y: i32, window: &WindowInfo) -> (i32, i32) {
        (window.bounds.x + local_x, window.bounds.y + local_y)
    }

    /// Adjusts an element's bounding rectangle when its parent window moves or resizes.
    pub fn adjust_for_window_movement(
        element_rect: &Rect,
        old_window: &WindowInfo,
        new_window: &WindowInfo,
    ) -> Rect {
        // Element bounds are stored window-local; if DPI scale changed, rescale
        if (old_window.display_scale - new_window.display_scale).abs() > 0.001 {
            let ratio = new_window.display_scale / old_window.display_scale;
            Rect {
                x: ((element_rect.x as f32) * ratio).round() as i32,
                y: ((element_rect.y as f32) * ratio).round() as i32,
                width: ((element_rect.width as f32) * ratio).round().max(1.0) as u32,
                height: ((element_rect.height as f32) * ratio).round().max(1.0) as u32,
            }
        } else {
            *element_rect
        }
    }
}
