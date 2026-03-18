import SwiftUI

/// Home screen displaying a greeting, navigation buttons, and available workout programs.
struct HomeView: View {
    @EnvironmentObject private var authService: AuthService
    @EnvironmentObject private var profileService: ProfileService
    @StateObject private var viewModel = HomeViewModel()

    /// Resolves the display name: profile name first, then auth display name, then fallback.
    private var displayName: String {
        let fromProfile = profileService.currentProfile?.name.trimmingCharacters(in: .whitespacesAndNewlines)
        if let name = fromProfile, !name.isEmpty { return name }
        return authService.displayName ?? viewModel.userName
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    HStack {
                        VStack(alignment: .leading) {
                            Text("Hi, \(displayName)")
                                .font(.title)
                                .fontWeight(.bold)
                                .foregroundColor(Color(.white))

                            Text("Let's start your fitness journey.")
                                .font(.subheadline)
                                .foregroundColor(.gray)
                        }
                        Spacer()
                        NavigationLink(destination: EditProfileView()) {
                            Image(systemName: "person.circle.fill")
                                .font(.largeTitle)
                                .foregroundColor(.white)
                        }
                    }

                    HStack(spacing: 20) {
                        MenuButton(icon: "flame.fill", title: "Workouts")
                        NavigationLink(destination: ProgressView()) {
                            MenuButton(icon: "chart.bar.fill", title: "Progress")
                        }

                    }

                    Text("Workout Programs")
                        .font(.headline)
                        .foregroundColor(.white)

                    VStack(spacing: 16) {
                        ForEach(viewModel.programs) { program in
                            WorkoutSetCard(
                                title: program.title,
                                description: program.description,
                                imageName: program.imageName
                            )
                        }
                    }

                    Spacer()
                }
                .padding()
            }
            .navigationBarHidden(true)
            .background(Color.black.edgesIgnoringSafeArea(.all))
        }
    }
}

// MARK: - Menu Button

/// Circular icon button with a label used for top navigation shortcuts.
fileprivate struct MenuButton: View {
    let icon: String
    let title: String
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title)
                .foregroundColor(.yellow)
                .frame(width: 60, height: 60)
                .background(Color.gray.opacity(0.2))
                .clipShape(Circle())
            Text(title)
                .foregroundColor(.white)
                .font(.footnote)
        }
    }
}

// MARK: - Workout Set Card

/// Card displaying a workout program with a background image, title, and description.
fileprivate struct WorkoutSetCard: View {
    let title: String
    let description: String
    let imageName: String

    var body: some View {
        NavigationLink(destination: WorkoutDetailView(setTitle: title)) {
            ZStack(alignment: .bottomLeading) {
                Image(imageName)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(height: 180)
                    .cornerRadius(16)
                    .clipped()
                
                LinearGradient(gradient: Gradient(colors: [.black.opacity(0.8), .clear]),
                               startPoint: .bottom, endPoint: .center)
                    .cornerRadius(16)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                    Text(description)
                        .font(.subheadline)
                        .foregroundColor(.gray)
                }
                .padding()
            }
        }
    }
}

#Preview {
    HomeView()
        .environmentObject(AuthService())
        .environmentObject(ProfileService())
}
