import Foundation
import SwiftUI
import AuthenticationServices

/// Handles authentication actions (Google, Apple) for the login screen.
@MainActor
final class LoginViewModel: ObservableObject {
    @Published var errorMessage: String?
    @Published var isLoading = false

    private let authService: AuthService

    init(authService: AuthService) {
        self.authService = authService
    }

    /// Exposes the auth service for UI binding (e.g., loading state).
    var auth: AuthService { authService }

    /// Initiates Google Sign-In and surfaces any resulting error.
    func signInWithGoogle() async {
        await authService.signInWithGoogle()
        errorMessage = authService.errorMessage
    }

    /// Handles the Apple Sign-In authorization result.
    func signInWithApple(authorization: ASAuthorization) async {
        await authService.signInWithApple(authorization: authorization)
        errorMessage = authService.errorMessage
    }

    /// Prepares the nonce for Sign in with Apple.
    func prepareAppleSignIn() {
        _ = authService.prepareAppleSignIn()
    }

    /// Clears displayed errors in both the ViewModel and AuthService.
    func clearError() {
        errorMessage = nil
        authService.clearError()
    }
}
