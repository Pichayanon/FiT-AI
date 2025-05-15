import Foundation

func loadMockSequence(from filename: String) -> [[Float]] {
    guard let url = Bundle.main.url(forResource: filename, withExtension: "json"),
          let data = try? Data(contentsOf: url),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let sequence = json["sequence"] as? [[Double]] else {
        print("Failed to load or parse \(filename).json")
        return []
    }
    return sequence.map { $0.map { Float($0) } }
}

let mockSequences: [[[Float]]] = [
    loadMockSequence(from: "squat_correct_1"),
    loadMockSequence(from: "squat_correct_2"),
    loadMockSequence(from: "squat_incorrect_1"),
    loadMockSequence(from: "squat_incorrect_2"),
]
