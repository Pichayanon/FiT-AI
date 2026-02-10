import SwiftUI
import FirebaseCore

@main
struct frontendApp: App {
    @StateObject private var authService = AuthService()

    init() {
        FirebaseApp.configure()
    }

    var body: some Scene {
        WindowGroup {
            Group {
                if authService.currentUser == nil {
                    LoginView(authService: authService)
                } else {
                    HomeView(authService: authService)
                }
            }
        }
    }
}
