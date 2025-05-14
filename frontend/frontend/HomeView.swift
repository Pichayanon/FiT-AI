import SwiftUI

struct HomeView: View {
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    HStack {
                        VStack(alignment: .leading) {
                            Text("Hi, Pichayanon")
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
                        WorkoutSetCard(
                            title: "Beginner Level 1",
                            description: "Squat, High Knees, Mountain Climbers",
                            imageName: "set1"
                        )
                        WorkoutSetCard(
                            title: "Beginner Level 2",
                            description: "Lunges, Plank, Jumping Jacks",
                            imageName: "set2"
                        )
                        WorkoutSetCard(
                            title: "Intermediate Level 1",
                            description: "Burpees, Push-Ups, Jump Squats",
                            imageName: "set3"
                        )
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

struct MenuButton: View {
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

struct WorkoutSetCard: View {
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
}
