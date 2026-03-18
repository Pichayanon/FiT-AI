import SwiftUI

/// Profile editing screen with form fields for name, age, weight, height, and sign-out.
struct EditProfileView: View {
    @EnvironmentObject private var authService: AuthService
    @EnvironmentObject private var profileService: ProfileService
    @StateObject private var viewModel = EditProfileViewModel()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                Image(systemName: "person.circle.fill")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 100, height: 100)
                    .foregroundColor(.yellow)
                    .padding(.top, 8)

                if let message = profileService.errorMessage {
                    Text(message)
                        .font(.footnote)
                        .foregroundColor(.red)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }

                VStack(alignment: .leading, spacing: 16) {
                    ProfileTextField(title: "Name", text: $viewModel.name, isNumber: false)
                    ProfileTextField(title: "Age", text: $viewModel.age, isNumber: true)
                    ProfileTextField(title: "Weight (kg)", text: $viewModel.weight, isNumber: true)
                    ProfileTextField(title: "Height (cm)", text: $viewModel.height, isNumber: true)
                }
                .padding(.horizontal, 4)

                Button(action: {
                    Task { await viewModel.save(using: profileService) }
                }) {
                    Group {
                        if profileService.isSaving {
                            SwiftUI.ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .black))
                        } else {
                            Text("Save")
                                .fontWeight(.semibold)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 52)
                    .background(Color.yellow)
                    .foregroundColor(.black)
                    .cornerRadius(14)
                }
                .buttonStyle(.plain)
                .disabled(profileService.isSaving)

                Button(action: {
                    authService.signOut()
                    dismiss()
                }) {
                    Text("Sign out")
                        .font(.body)
                        .frame(maxWidth: .infinity)
                        .frame(height: 52)
                        .foregroundColor(.red)
                }
                .buttonStyle(.plain)
                .padding(.bottom, 32)
            }
            .padding(.horizontal, 20)
        }
        .scrollDismissesKeyboard(.interactively)
        .background(Color.black.ignoresSafeArea())
        .navigationTitle("Edit Profile")
        .navigationBarTitleDisplayMode(.inline)
        .foregroundColor(.white)
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") {
                    hideKeyboard()
                }
                .foregroundColor(.yellow)
            }
        }
        .onAppear {
            viewModel.fillFromProfile(profileService.currentProfile)
        }
    }

    private func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}

// MARK: - Profile Text Field

/// Styled text field with a label, supporting both text and numeric keyboard types.
fileprivate struct ProfileTextField: View {
    let title: String
    @Binding var text: String
    var isNumber: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.subheadline)
                .foregroundColor(.gray)
            TextField(title, text: $text)
                .padding(.horizontal, 16)
                .padding(.vertical, 14)
                .background(Color.gray.opacity(0.25))
                .cornerRadius(12)
                .keyboardType(isNumber ? .decimalPad : .default)
                .autocorrectionDisabled()
        }
    }
}
