import SwiftUI

struct EditProfileView: View {
    @ObservedObject var authService: AuthService
    @ObservedObject var profileService: ProfileService
    @StateObject private var viewModel: EditProfileViewModel
    @Environment(\.dismiss) private var dismiss

    init(authService: AuthService, profileService: ProfileService) {
        self.authService = authService
        self.profileService = profileService
        _viewModel = StateObject(wrappedValue: EditProfileViewModel(profileService: profileService))
    }

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

                // ปุ่ม Save ความสูงคงที่ ไม่กระตุก
                Button(action: {
                    Task { await viewModel.save() }
                }) {
                    HStack(spacing: 8) {
                        if profileService.isSaving {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .black))
                        }
                        Text(profileService.isSaving ? "Saving..." : "Save")
                            .fontWeight(.semibold)
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
