import Foundation
import SwiftUI
import CryptoKit
import FirebaseCore
import FirebaseAuth
import GoogleSignIn
import AuthenticationServices

/// Manages Firebase Auth + OAuth (Google, Sign in with Apple). Configure Firebase in App init.
@MainActor
final class AuthService: ObservableObject {
    @Published private(set) var currentUser: User?
    @Published private(set) var isLoading = false
    @Published var errorMessage: String?

    private var authStateHandler: AuthStateDidChangeListenerHandle?

    /// Set this before starting Sign in with Apple; used when credential returns.
    var currentAppleNonce: String?

    /// Use this as `request.nonce` when creating ASAuthorizationAppleIDRequest.
    var hashedNonceForAppleSignIn: String? {
        guard let nonce = currentAppleNonce else { return nil }
        return sha256(nonce)
    }

    init() {
        authStateHandler = Auth.auth().addStateDidChangeListener { [weak self] _, user in
            Task { @MainActor in
                self?.currentUser = user
            }
        }
        currentUser = Auth.auth().currentUser
    }

    deinit {
        if let handler = authStateHandler {
            Auth.auth().removeStateDidChangeListener(handler)
        }
    }

    var isLoggedIn: Bool { currentUser != nil }
    var displayName: String? { currentUser?.displayName ?? currentUser?.email?.components(separatedBy: "@").first }

    // MARK: - Google Sign-In

    func signInWithGoogle() async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }

        guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let rootVC = windowScene.windows.first?.rootViewController else {
            errorMessage = "Could not get root view controller"
            return
        }

        do {
            let clientID = FirebaseApp.app()?.options.clientID ?? ""
            let config = GIDConfiguration(clientID: clientID)
            GIDSignIn.sharedInstance.configuration = config
            let result = try await GIDSignIn.sharedInstance.signIn(withPresenting: rootVC)

            guard let idToken = result.user.idToken?.tokenString else {
                errorMessage = "Google ID token is missing"
                return
            }

            let credential = GoogleAuthProvider.credential(withIDToken: idToken, accessToken: result.user.accessToken.tokenString)
            _ = try await Auth.auth().signIn(with: credential)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: - Sign in with Apple

    /// Call before presenting Apple ID request. Use returned nonce in ASAuthorizationAppleIDRequest.
    func prepareAppleSignIn() -> String {
        let nonce = randomNonceString()
        currentAppleNonce = nonce
        return nonce
    }

    func signInWithApple(authorization: ASAuthorization) async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false; currentAppleNonce = nil }

        guard let nonce = currentAppleNonce else {
            errorMessage = "Apple sign-in nonce missing"
            return
        }
        guard let appleIDCredential = authorization.credential as? ASAuthorizationAppleIDCredential,
              let idTokenData = appleIDCredential.identityToken,
              let idToken = String(data: idTokenData, encoding: .utf8) else {
            errorMessage = "Invalid Apple credential"
            return
        }

        let credential = OAuthProvider.appleCredential(withIDToken: idToken, rawNonce: nonce, fullName: appleIDCredential.fullName)

        do {
            _ = try await Auth.auth().signIn(with: credential)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func randomNonceString(length: Int = 32) -> String {
        precondition(length > 0)
        let charset: [Character] = Array("0123456789ABCDEFGHIJKLMNOPQRSTUVXYZabcdefghijklmnopqrstuvwxyz-._")
        var result = ""
        var remainingLength = length
        while remainingLength > 0 {
            let randoms: [UInt8] = (0 ..< 16).map { _ in UInt8.random(in: 0 ..< UInt8.max) }
            for random in randoms {
                if remainingLength == 0 { break }
                if random < charset.count {
                    result.append(charset[Int(random)])
                    remainingLength -= 1
                }
            }
        }
        return result
    }

    private func sha256(_ string: String) -> String {
        let data = Data(string.utf8)
        let hash = SHA256.hash(data: data)
        return hash.map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Sign out

    func signOut() {
        errorMessage = nil
        do {
            try Auth.auth().signOut()
            GIDSignIn.sharedInstance.signOut()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func clearError() {
        errorMessage = nil
    }
}
