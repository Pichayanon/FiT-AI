import SwiftUI

/// Login screen with Google Sign-In (and future Apple Sign-In support).
struct LoginView: View {
    @StateObject private var viewModel: LoginViewModel

    init(authService: AuthService) {
        _viewModel = StateObject(wrappedValue: LoginViewModel(authService: authService))
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 32) {
                Spacer()

                VStack(spacing: 12) {
                    Image(systemName: "figure.run")
                        .font(.system(size: 64))
                        .foregroundColor(.yellow)
                    Text("FIT-AI")
                        .font(.largeTitle.bold())
                        .foregroundColor(.white)
                    Text("Sign in to start your workout")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                }

                Spacer()

                VStack(spacing: 16) {
                    if viewModel.auth.isLoading {
                        SwiftUI.ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(1.2)
                            .padding()
                    } else {
                        Button {
                            Task { await viewModel.signInWithGoogle() }
                        } label: {
                            HStack(spacing: 12) {
                                Image(systemName: "globe")
                                    .font(.title3)
                                Text("Continue with Google")
                                    .font(.headline)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(Color.white)
                            .foregroundColor(.black)
                            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                        }
                        .disabled(viewModel.auth.isLoading)
                    }

                    if let message = viewModel.errorMessage {
                        Text(message)
                            .font(.caption)
                            .foregroundColor(.red)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                }
                .padding(.horizontal, 24)

                Spacer()
                    .frame(height: 48)
            }
        }
        .onTapGesture {
            viewModel.clearError()
        }
    }
}

#Preview {
    LoginView(authService: AuthService())
}
