import SwiftUI
import AVKit

/// Represents one playback request for a mistake inside a recorded workout session.
struct SessionMistakePlayback: Identifiable {
    let id = UUID()
    let title: String
    let subtitle: String?
    let videoURL: URL
    let atSecond: Int
}

/// Full-screen player sheet that seeks to the selected mistake timestamp on open.
struct SessionVideoPlaybackView: View {
    let playback: SessionMistakePlayback

    @Environment(\.dismiss) private var dismiss
    @State private var player: AVPlayer

    init(playback: SessionMistakePlayback) {
        self.playback = playback
        _player = State(initialValue: AVPlayer(url: playback.videoURL))
    }

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color.black, Color(red: 0.08, green: 0.06, blue: 0.02)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            Circle()
                .fill(Color.orange.opacity(0.16))
                .frame(width: 240, height: 240)
                .blur(radius: 28)
                .offset(x: 130, y: -260)

            Circle()
                .fill(Color.yellow.opacity(0.10))
                .frame(width: 220, height: 220)
                .blur(radius: 32)
                .offset(x: -150, y: 260)

            VStack(spacing: 20) {
                header

                VStack(alignment: .leading, spacing: 18) {
                    playerCard

                    VStack(alignment: .leading, spacing: 10) {
                        Text(playback.title)
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(.white)

                        if let subtitle = playback.subtitle {
                            Text(subtitle)
                                .font(.subheadline)
                                .foregroundColor(.gray)
                        }

                        Text("We jump straight to the selected mistake moment so you can review your form immediately.")
                            .font(.subheadline)
                            .foregroundColor(.white.opacity(0.78))
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    if let subtitle = playback.subtitle {
                        ReplayInfoChip(
                            icon: "figure.strengthtraining.traditional",
                            label: "Context",
                            value: subtitle
                        )
                    }
                }
                .padding(18)
                .background(
                    RoundedRectangle(cornerRadius: 30, style: .continuous)
                        .fill(Color.white.opacity(0.06))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 30, style: .continuous)
                        .stroke(Color.white.opacity(0.08), lineWidth: 1)
                )
                .shadow(color: Color.black.opacity(0.35), radius: 24, x: 0, y: 16)

                Spacer(minLength: 0)
            }
            .padding(.horizontal, 20)
            .padding(.top, 20)
            .padding(.bottom, 28)
        }
        .onAppear {
            startPlayback()
        }
        .onDisappear {
            player.pause()
        }
    }

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Mistake Replay")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundColor(.white)

                Text("See exactly where the form started to break.")
                    .font(.subheadline)
                    .foregroundColor(.gray)
            }

            Spacer()

            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(.white)
                    .frame(width: 36, height: 36)
                    .background(Color.white.opacity(0.10))
                    .clipShape(Circle())
            }
            .buttonStyle(.plain)
        }
    }

    private var playerCard: some View {
        ZStack(alignment: .topLeading) {
            VideoPlayer(player: player)
                .frame(maxWidth: .infinity)
                .frame(height: 340)
                .background(Color.black)
                .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .stroke(Color.white.opacity(0.08), lineWidth: 1)
                )

            ReplayInfoChip(
                icon: "play.circle.fill",
                label: "Replay",
                value: "Selected mistake"
            )
            .padding(14)
        }
    }

    private func startPlayback() {
        let targetSecond = max(0, playback.atSecond - 1)
        let targetTime = CMTime(seconds: Double(targetSecond), preferredTimescale: 600)
        player.seek(to: targetTime, toleranceBefore: .zero, toleranceAfter: .zero) { _ in
            player.play()
        }
    }
}

private struct ReplayInfoChip: View {
    let icon: String
    let label: String
    let value: String

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .font(.subheadline)
                .foregroundColor(.yellow)

            VStack(alignment: .leading, spacing: 2) {
                Text(label.uppercased())
                    .font(.caption2)
                    .fontWeight(.bold)
                    .foregroundColor(.gray)
                Text(value)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                    .lineLimit(1)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(Color.black.opacity(0.36))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Color.white.opacity(0.06), lineWidth: 1)
        )
    }
}
