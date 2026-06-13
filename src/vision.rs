use crate::llm::ImageAttachment;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use std::collections::BTreeSet;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const MAX_IMAGES_PER_MESSAGE: usize = 8;
const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum VisionError {
    #[error("image file is too large: {path} is {size} bytes, max is {max} bytes")]
    ImageTooLarge {
        path: String,
        size: usize,
        max: usize,
    },
    #[error("failed to read image {path}: {source}")]
    ReadImage {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to access clipboard: {0}")]
    Clipboard(String),
    #[error("failed to encode clipboard image: {0}")]
    ClipboardEncode(String),
}

pub async fn attachments_from_text(input: &str) -> Result<Vec<ImageAttachment>, VisionError> {
    let mut attachments = Vec::new();
    let mut seen = BTreeSet::new();

    for candidate in path_candidates(input) {
        if attachments.len() >= MAX_IMAGES_PER_MESSAGE {
            break;
        }
        if let Some(attachment) = attachment_from_data_url(&candidate) {
            if seen.insert(attachment.source.clone()) {
                attachments.push(attachment);
            }
            continue;
        }

        let Some((path, mime_type)) = image_path_candidate(&candidate) else {
            continue;
        };
        let canonical = std::fs::canonicalize(&path).unwrap_or(path.clone());
        let key = canonical.display().to_string();
        if !seen.insert(key) {
            continue;
        }
        attachments.push(read_image_attachment(canonical, mime_type).await?);
    }

    Ok(attachments)
}

pub async fn attachments_from_paths(paths: &[String]) -> Result<Vec<ImageAttachment>, VisionError> {
    let mut attachments = Vec::new();
    let mut seen = BTreeSet::new();

    for path in paths {
        if attachments.len() >= MAX_IMAGES_PER_MESSAGE {
            break;
        }
        let Some((path, mime_type)) = image_path_candidate(path) else {
            continue;
        };
        let canonical = std::fs::canonicalize(&path).unwrap_or(path.clone());
        let key = canonical.display().to_string();
        if !seen.insert(key) {
            continue;
        }
        attachments.push(read_image_attachment(canonical, mime_type).await?);
    }

    Ok(attachments)
}

async fn read_image_attachment(
    path: PathBuf,
    mime_type: &'static str,
) -> Result<ImageAttachment, VisionError> {
    let bytes = tokio::fs::read(&path)
        .await
        .map_err(|source| VisionError::ReadImage {
            path: path.display().to_string(),
            source,
        })?;
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(VisionError::ImageTooLarge {
            path: path.display().to_string(),
            size: bytes.len(),
            max: MAX_IMAGE_BYTES,
        });
    }
    Ok(ImageAttachment::new(
        path.display().to_string(),
        mime_type,
        STANDARD.encode(bytes),
    ))
}

pub fn image_from_clipboard() -> Result<Option<ImageAttachment>, VisionError> {
    let mut clipboard =
        arboard::Clipboard::new().map_err(|error| VisionError::Clipboard(error.to_string()))?;
    match clipboard.get_image() {
        Ok(image) => encode_clipboard_image(image).map(Some),
        Err(arboard::Error::ContentNotAvailable) => Ok(None),
        Err(error) => Err(VisionError::Clipboard(error.to_string())),
    }
}

pub fn text_from_clipboard() -> Result<Option<String>, VisionError> {
    let mut clipboard =
        arboard::Clipboard::new().map_err(|error| VisionError::Clipboard(error.to_string()))?;
    match clipboard.get_text() {
        Ok(text) => Ok(Some(text)),
        Err(arboard::Error::ContentNotAvailable) => Ok(None),
        Err(error) => Err(VisionError::Clipboard(error.to_string())),
    }
}

fn encode_clipboard_image(image: arboard::ImageData<'_>) -> Result<ImageAttachment, VisionError> {
    let width = image.width as u32;
    let height = image.height as u32;
    let rgba = image::RgbaImage::from_raw(width, height, image.bytes.into_owned())
        .ok_or_else(|| VisionError::ClipboardEncode("invalid RGBA clipboard buffer".to_string()))?;
    let mut encoded = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(rgba)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .map_err(|error| VisionError::ClipboardEncode(error.to_string()))?;
    Ok(ImageAttachment::new(
        "clipboard",
        "image/png",
        STANDARD.encode(encoded.into_inner()),
    ))
}

fn attachment_from_data_url(candidate: &str) -> Option<ImageAttachment> {
    let candidate = candidate.trim();
    let rest = candidate.strip_prefix("data:image/")?;
    let (subtype, data) = rest.split_once(";base64,")?;
    if subtype.is_empty() || data.is_empty() {
        return None;
    }
    Some(ImageAttachment::new(
        "pasted data URL",
        format!("image/{subtype}"),
        data.to_string(),
    ))
}

fn image_path_candidate(candidate: &str) -> Option<(PathBuf, &'static str)> {
    let path = normalize_path_candidate(candidate)?;
    let mime_type = mime_type_for_path(&path)?;
    path.is_file().then_some((path, mime_type))
}

fn normalize_path_candidate(candidate: &str) -> Option<PathBuf> {
    let trimmed = candidate.trim().trim_matches(|ch: char| {
        matches!(
            ch,
            '\'' | '"' | '`' | '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | ',' | ';'
        )
    });
    let trimmed = trimmed.strip_prefix("file://").unwrap_or(trimmed);
    if trimmed.is_empty() {
        return None;
    }
    if let Some(rest) = trimmed.strip_prefix("~/")
        && let Some(home) = directories::BaseDirs::new().map(|dirs| dirs.home_dir().to_path_buf())
    {
        return Some(home.join(rest));
    }
    Some(PathBuf::from(trimmed))
}

fn mime_type_for_path(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "webp" => Some("image/webp"),
        "gif" => Some("image/gif"),
        _ => None,
    }
}

fn path_candidates(input: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    let mut escaped = false;

    for ch in input.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if let Some(active_quote) = quote {
            if ch == active_quote {
                quote = None;
                candidates.push(current.clone());
                current.clear();
            } else {
                current.push(ch);
            }
            continue;
        }
        if matches!(ch, '\'' | '"') {
            if !current.is_empty() {
                candidates.push(current.clone());
                current.clear();
            }
            quote = Some(ch);
            continue;
        }
        if ch.is_whitespace() {
            if !current.is_empty() {
                candidates.push(current.clone());
                current.clear();
            }
            continue;
        }
        current.push(ch);
    }
    if !current.is_empty() {
        candidates.push(current);
    }
    candidates
}
