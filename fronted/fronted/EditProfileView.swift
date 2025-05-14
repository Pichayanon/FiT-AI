import SwiftUI

struct EditProfileView: View {
    @State private var name: String = "Pichayanon"
    @State private var age: String = "21"
    @State private var weight: String = "70"
    @State private var height: String = "175"
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "person.circle.fill")
                .resizable()
                .frame(width: 120, height: 120)
                .foregroundColor(.yellow)
                .padding(.top, 40)
            
            Group {
                ProfileTextField(title: "Name", text: $name)
                ProfileTextField(title: "Age", text: $age)
                ProfileTextField(title: "Weight (kg)", text: $weight)
                ProfileTextField(title: "Height (cm)", text: $height)
            }
            
            Spacer()
            
            Button(action: {
                print("Saved: \(name), \(age), \(weight), \(height)")
            }) {
                Text("Save")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.yellow)
                    .foregroundColor(.black)
                    .cornerRadius(20)
            }
            .padding(.bottom)
        }
        .padding(.horizontal)
        .background(Color.black.edgesIgnoringSafeArea(.all))
        .navigationTitle("Edit Profile")
        .foregroundColor(.white)
    }
}

struct ProfileTextField: View {
    let title: String
    @Binding var text: String
    
    var body: some View {
        VStack(alignment: .leading) {
            Text(title)
                .foregroundColor(.gray)
            TextField(title, text: $text)
                .padding()
                .background(Color.gray.opacity(0.2))
                .cornerRadius(12)
                .keyboardType(title == "Age" || title.contains("Weight") || title.contains("Height") ? .numberPad : .default)
        }
    }
}
