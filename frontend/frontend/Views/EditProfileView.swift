import SwiftUI

struct EditProfileView: View {
    @StateObject private var viewModel = EditProfileViewModel()
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "person.circle.fill")
                .resizable()
                .frame(width: 120, height: 120)
                .foregroundColor(.yellow)
                .padding(.top, 40)
            
            Group {
                ProfileTextField(title: "Name", text: $viewModel.name)
                ProfileTextField(title: "Age", text: $viewModel.age)
                ProfileTextField(title: "Weight (kg)", text: $viewModel.weight)
                ProfileTextField(title: "Height (cm)", text: $viewModel.height)
            }
            
            Spacer()
            
            Button(action: {
                viewModel.save()
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

fileprivate struct ProfileTextField: View {
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
