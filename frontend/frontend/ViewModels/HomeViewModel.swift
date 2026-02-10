import Foundation
import SwiftUI

@MainActor
final class HomeViewModel: ObservableObject {
    @Published var userName: String = "Pichayanon"

    struct WorkoutProgram: Identifiable {
        let id = UUID()
        let title: String
        let description: String
        let imageName: String
    }

    let programs: [WorkoutProgram] = [
        .init(
            title: "Beginner Level 1",
            description: "Squat, High Knees, Mountain Climbers",
            imageName: "set1"
        ),
        .init(
            title: "Beginner Level 2",
            description: "Lunges, Plank, Jumping Jacks",
            imageName: "set2"
        ),
        .init(
            title: "Intermediate Level 1",
            description: "Burpees, Push-Ups, Jump Squats",
            imageName: "set3"
        )
    ]
}

