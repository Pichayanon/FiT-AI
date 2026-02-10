import Foundation
import SwiftUI
import AuthenticationServices

@MainActor
final class LoginViewModel: ObservableObject {
    @Published var errorMessage: String?
    @Published var isLoading = false

    private let authService: AuthService

    init(authService: AuthService) {
        self.authService = authService
    }

    var auth: AuthService { authService }

    func signInWithGoogle() async {
        await authService.signInWithGoogle()
        errorMessage = authService.errorMessage
    }

    func signInWithApple(authorization: ASAuthorization) async {
        await authService.signInWithApple(authorization: authorization)
        errorMessage = authService.errorMessage
    }

    func prepareAppleSignIn() {
        _ = authService.prepareAppleSignIn()
    }

    func clearError() {
        errorMessage = nil
        authService.clearError()
    }
}
