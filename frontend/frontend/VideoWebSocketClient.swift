import Foundation

final class VideoWebSocketClient: NSObject {
    struct Config {
        let url: URL
        let connectTimeout: TimeInterval = 5
    }

    private let config: Config
    private var task: URLSessionWebSocketTask?
    private var session: URLSession?

    var onConnected: (() -> Void)?
    var onDisconnected: ((Error?) -> Void)?
    var onTextMessage: ((String) -> Void)?

    init(config: Config) {
        self.config = config
        super.init()
    }

    func connect() {
        disconnect()

        let conf = URLSessionConfiguration.default
        conf.timeoutIntervalForRequest = config.connectTimeout
        let session = URLSession(configuration: conf, delegate: nil, delegateQueue: nil)
        self.session = session

        let task = session.webSocketTask(with: config.url)
        self.task = task
        task.resume()

        onConnected?()
        receiveLoop()
    }

    func disconnect() {
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        session?.invalidateAndCancel()
        session = nil
    }

    func sendText(_ text: String) {
        task?.send(.string(text)) { err in
            if let err { print("WS send error:", err) }
        }
    }

    private func receiveLoop() {
        task?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure(let err):
                self.onDisconnected?(err)
            case .success(let msg):
                switch msg {
                case .string(let s):
                    self.onTextMessage?(s)
                case .data(let d):
                    if let s = String(data: d, encoding: .utf8) {
                        self.onTextMessage?(s)
                    }
                @unknown default:
                    break
                }
                self.receiveLoop()
            }
        }
    }
}
