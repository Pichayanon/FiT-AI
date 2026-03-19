import SwiftUI
import AVKit

/// Represents one playback request for a mistake inside a recorded workout session.
struct SessionMistakePlayback: Identifiable {
    let id = UUID()
    let title: String
    let subtitle: String?
    let videoURL: URL
    let atSecond: Double
    let clipStartSecond: Double
    let clipDurationSeconds: Double

    init(
        title: String,
        subtitle: String?,
        videoURL: URL,
        atSecond: Double,
        leadingPadding: Double = 2.0,
        trailingPadding: Double = 1.5
    ) {
        self.title = title
        self.subtitle = subtitle
        self.videoURL = videoURL
        self.atSecond = atSecond
        self.clipStartSecond = max(0, atSecond - leadingPadding)
        self.clipDurationSeconds = max(2.5, leadingPadding + trailingPadding)
    }
}

/// Full-screen player sheet that seeks to the selected mistake timestamp on open.
struct SessionVideoPlaybackView: View {
    let playback: SessionMistakePlayback

    @Environment(\.dismiss) private var dismiss
    @State private var player: AVPlayer

    init(playback: SessionMistakePlayback) {
        self.playback = playback
        _player = State(initialValue: Self.makePlayer(for: playback))
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

                        Text(mistakeExplanation)
                            .font(.subheadline)
                            .foregroundColor(.white.opacity(0.78))
                            .fixedSize(horizontal: false, vertical: true)
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
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.title3)
                        .foregroundColor(.yellow)

                    Text("Mistake Replay")
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                }

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
        VideoPlayer(player: player)
            .frame(maxWidth: .infinity)
            .frame(height: 340)
            .background(Color.black)
            .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .stroke(Color.white.opacity(0.08), lineWidth: 1)
            )
    }

    private func startPlayback() {
        let clipDuration = playback.clipDurationSeconds
        let currentDuration = player.currentItem?.duration.seconds ?? 0
        let startSecond = currentDuration > clipDuration + 0.25 ? playback.clipStartSecond : 0
        let startTime = CMTime(seconds: startSecond, preferredTimescale: 600)

        player.seek(to: startTime, toleranceBefore: .zero, toleranceAfter: .zero) { _ in
            player.play()
        }
    }

    private static func makePlayer(for playback: SessionMistakePlayback) -> AVPlayer {
        let asset = AVURLAsset(url: playback.videoURL)
        let composition = AVMutableComposition()
        let startTime = CMTime(seconds: playback.clipStartSecond, preferredTimescale: 600)
        let duration = CMTime(seconds: playback.clipDurationSeconds, preferredTimescale: 600)
        let timeRange = CMTimeRange(start: startTime, duration: duration)

        var didInsertVideo = false

        if let sourceVideoTrack = asset.tracks(withMediaType: .video).first,
           let compositionVideoTrack = composition.addMutableTrack(
               withMediaType: .video,
               preferredTrackID: kCMPersistentTrackID_Invalid
           ) {
            do {
                try compositionVideoTrack.insertTimeRange(timeRange, of: sourceVideoTrack, at: .zero)
                compositionVideoTrack.preferredTransform = sourceVideoTrack.preferredTransform
                didInsertVideo = true
            } catch {
                print("[SessionVideoPlaybackView] Video clip build error: \(error)")
            }
        }

        if let sourceAudioTrack = asset.tracks(withMediaType: .audio).first,
           let compositionAudioTrack = composition.addMutableTrack(
               withMediaType: .audio,
               preferredTrackID: kCMPersistentTrackID_Invalid
           ) {
            do {
                try compositionAudioTrack.insertTimeRange(timeRange, of: sourceAudioTrack, at: .zero)
            } catch {
                print("[SessionVideoPlaybackView] Audio clip build error: \(error)")
            }
        }

        if didInsertVideo {
            let item = AVPlayerItem(asset: composition)
            let player = AVPlayer(playerItem: item)
            player.actionAtItemEnd = .pause
            return player
        }

        let fallbackItem = AVPlayerItem(url: playback.videoURL)
        fallbackItem.forwardPlaybackEndTime = CMTime(
            seconds: playback.clipStartSecond + playback.clipDurationSeconds,
            preferredTimescale: 600
        )
        let player = AVPlayer(playerItem: fallbackItem)
        player.actionAtItemEnd = .pause
        player.seek(to: CMTime(seconds: playback.clipStartSecond, preferredTimescale: 600))
        return player
    }
    
    private var mistakeExplanation: String {
        let reason = playback.title.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let context = playback.subtitle?.lowercased() ?? ""

        if reason.contains("feet too close") {
            if context.contains("wall-sit") {
                return "Your feet are too close to the wall here, which makes the sitting angle and weight distribution less stable. Step your feet slightly farther away and keep your back supported against the wall."
            }
            return "Your stance is too narrow here. Move your feet slightly wider so you have a more stable base and better leg alignment."
        }

        if reason.contains("knees in") || reason.contains("knee in") {
            return "Your knees are collapsing inward here. Push them out in line with your toes and keep your weight balanced on both sides."
        }

        if reason.contains("round back") {
            return "Your back is rounding too much here. Brace your core, keep your chest open, and maintain a more neutral spine throughout the movement."
        }

        if reason.contains("torso lean forward") {
            return "Your torso is leaning too far forward here. Lift your chest, brace your core, and keep your weight more centered over your feet."
        }

        if reason.contains("knee over toe") {
            return "Your knee is moving too far past your toes here. Send your hips back more and avoid driving your shin too far forward."
        }

        if reason.contains("not deep enough") {
            return "You are not going deep enough here. Lower your hips a bit more while keeping your chest up and your knees steady."
        }

        if reason.contains("stand too narrow") {
            return "Your setup stance is too narrow here. Move your feet slightly wider to create a more stable base."
        }

        if reason.contains("stand too wide") {
            return "Your setup stance is too wide here. Bring your feet in a little so the position is easier to control."
        }

        if reason.contains("hips too high") {
            return "Your hips are too high here. Lower them slightly so your body stays in a straighter line."
        }

        if reason.contains("low hips") || reason.contains("hips too low") {
            return "Your hips are dropping too low here. Tighten your core and lift them slightly so your body stays straighter."
        }

        if reason.contains("head too low") {
            return "Your head is too low here. Look slightly ahead on the floor and keep your neck aligned naturally with your torso."
        }

        if reason.contains("elbow") {
            return "Your elbow position is unstable here. Stay active through your arms and place your elbows in a position you can control more comfortably."
        }

        return "Review this moment closely to see where your form starts to drift, then slow that part down and make it more controlled."
    }
}
