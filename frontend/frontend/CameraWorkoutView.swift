import SwiftUI
import AVFoundation

struct CameraWorkoutView: View {
    @StateObject private var cameraManager = CameraManager()

    var body: some View {
        ZStack {
            CameraPreviewView(session: cameraManager.session)
                .edgesIgnoringSafeArea(.all)

            VStack {
                HStack {
                    Spacer()
                    VStack(alignment: .trailing) {
                        Text("Reps: \(cameraManager.reps)")
                            .foregroundColor(.white)
                        Text("Time: \(cameraManager.elapsedTime)s")
                            .foregroundColor(.white)
                    }
                    .padding()
                }

                Spacer()

                Text(cameraManager.feedback)
                    .padding()
                    .background(Color.black.opacity(0.7))
                    .foregroundColor(.yellow)
                    .cornerRadius(12)
            }
        }
        .onAppear {
            cameraManager.startSession()
        }
        .onDisappear {
            cameraManager.stopSession()
        }
    }
}
