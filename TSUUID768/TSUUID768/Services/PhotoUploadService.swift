import Foundation
import UIKit
import SwiftyDropbox

/// Uploads photos directly to Dropbox, returns a tinyurl-shortened direct link.
///
/// Flow:
///   1. Upload JPEG to /TSUUID768/photos/ via Dropbox API
///   2. Create a shared link, transform to dl.dropboxusercontent.com (direct download URL)
///   3. POST to tinyurl.com/api-create.php to get short URL
///   4. Return the 8-char tinyurl ID (e.g., "2xpvlcsu")
///
/// Bypasses iOS Photos entirely. The tinyurl ID becomes the canonical retrieval key.
@MainActor
final class PhotoUploadService {
    enum UploadError: Error, LocalizedError {
        case notAuthorized
        case uploadFailed(String)
        case shareLinkFailed(String)
        case tinyurlFailed(String)
        case jpegEncodingFailed

        var errorDescription: String? {
            switch self {
            case .notAuthorized: return "Not signed in to Dropbox"
            case .uploadFailed(let s): return "Upload failed: \(s)"
            case .shareLinkFailed(let s): return "Share link failed: \(s)"
            case .tinyurlFailed(let s): return "Tinyurl failed: \(s)"
            case .jpegEncodingFailed: return "Couldn't encode image as JPEG"
            }
        }
    }

    struct UploadResult {
        let dropboxPath: String
        let directURL: URL      // dl.dropboxusercontent.com direct link
        let tinyurl: URL        // https://tinyurl.com/xxxxxxxx
        let tinyurlId: String   // "xxxxxxxx" (8-char ID)
    }

    /// Full pipeline: UIImage → Dropbox → tinyurl
    func upload(_ image: UIImage, compressionQuality: CGFloat = 0.8) async throws -> UploadResult {
        guard let client = DropboxClientsManager.authorizedClient else {
            throw UploadError.notAuthorized
        }

        guard let jpegData = image.jpegData(compressionQuality: compressionQuality) else {
            throw UploadError.jpegEncodingFailed
        }

        // 1. Upload to Dropbox
        let timestamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "")
            .replacingOccurrences(of: "-", with: "")
        let dropboxPath = "/TSUUID768/photos/\(timestamp).jpg"
        let metadata = try await uploadData(jpegData, to: dropboxPath, client: client)

        // 2. Create shared link → transform to direct URL
        let directURL = try await createDirectLink(path: metadata.pathLower ?? dropboxPath, client: client)

        // 3. Shorten with tinyurl
        let (tinyurl, tinyurlId) = try await shortenURL(directURL)

        return UploadResult(
            dropboxPath: dropboxPath,
            directURL: directURL,
            tinyurl: tinyurl,
            tinyurlId: tinyurlId
        )
    }

    // MARK: - Dropbox upload

    private func uploadData(_ data: Data, to path: String, client: DropboxClient) async throws -> Files.FileMetadata {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Files.FileMetadata, Error>) in
            client.files.upload(path: path, mode: .overwrite, input: data)
                .response { result, error in
                    if let result = result {
                        continuation.resume(returning: result)
                    } else if let error = error {
                        continuation.resume(throwing: UploadError.uploadFailed("\(error)"))
                    }
                }
        }
    }

    // MARK: - Shared link → direct URL

    private func createDirectLink(path: String, client: DropboxClient) async throws -> URL {
        let shareURL = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, Error>) in
            client.sharing.createSharedLinkWithSettings(path: path)
                .response { result, error in
                    if let result = result {
                        continuation.resume(returning: result.url)
                    } else if let error = error {
                        // If link already exists, fetch it
                        client.sharing.listSharedLinks(path: path).response { listResult, listError in
                            if let listResult = listResult, let first = listResult.links.first {
                                continuation.resume(returning: first.url)
                            } else {
                                continuation.resume(throwing: UploadError.shareLinkFailed("\(error)"))
                            }
                        }
                    }
                }
        }

        // Transform www.dropbox.com/... to dl.dropboxusercontent.com/...
        let direct = shareURL
            .replacingOccurrences(of: "www.dropbox.com", with: "dl.dropboxusercontent.com")
            .replacingOccurrences(of: "?dl=0", with: "")
            .replacingOccurrences(of: "&dl=0", with: "")

        guard let url = URL(string: direct) else {
            throw UploadError.shareLinkFailed("Invalid URL: \(direct)")
        }
        return url
    }

    // MARK: - Tinyurl

    private func shortenURL(_ url: URL) async throws -> (URL, String) {
        let encoded = url.absoluteString.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
        guard let tinyurlAPI = URL(string: "https://tinyurl.com/api-create.php?url=\(encoded)") else {
            throw UploadError.tinyurlFailed("Bad API URL")
        }

        let (data, response) = try await URLSession.shared.data(from: tinyurlAPI)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw UploadError.tinyurlFailed("HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0)")
        }

        guard let text = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
              let shortURL = URL(string: text),
              text.hasPrefix("https://tinyurl.com/") else {
            throw UploadError.tinyurlFailed("Bad response: \(String(data: data, encoding: .utf8) ?? "")")
        }

        // Extract 8-char ID
        let id = String(text.dropFirst("https://tinyurl.com/".count))
        return (shortURL, id)
    }
}
