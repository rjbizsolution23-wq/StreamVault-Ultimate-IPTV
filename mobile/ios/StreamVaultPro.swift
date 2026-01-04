// StreamVault Pro - iOS Native Application
// Complete iOS app with streaming, authentication, and payments

import UIKit
import SwiftUI
import AVFoundation
import AVKit
import Combine
import Network

@main
struct StreamVaultProApp: App {
    @StateObject private var authManager = AuthenticationManager()
    @StateObject private var streamingManager = StreamingManager()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(authManager)
                .environmentObject(streamingManager)
                .onAppear {
                    setupApplication()
                }
        }
    }
    
    private func setupApplication() {
        // Configure streaming settings
        streamingManager.configure()
        
        // Check authentication status
        authManager.checkAuthStatus()
        
        // Setup analytics
        AnalyticsManager.shared.initialize()
    }
}

// MARK: - Main Content View
struct ContentView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    
    var body: some View {
        Group {
            if authManager.isAuthenticated {
                MainTabView()
            } else {
                AuthenticationView()
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)) { _ in
            authManager.refreshTokenIfNeeded()
        }
    }
}

// MARK: - Main Tab Navigation
struct MainTabView: View {
    var body: some View {
        TabView {
            LiveTVView()
                .tabItem {
                    Image(systemName: "tv")
                    Text("Live TV")
                }
            
            MoviesView()
                .tabItem {
                    Image(systemName: "film")
                    Text("Movies")
                }
            
            SportsView()
                .tabItem {
                    Image(systemName: "sportscourt")
                    Text("Sports")
                }
            
            MyLibraryView()
                .tabItem {
                    Image(systemName: "heart")
                    Text("My Library")
                }
            
            ProfileView()
                .tabItem {
                    Image(systemName: "person.circle")
                    Text("Profile")
                }
        }
        .accentColor(.streamVaultBlue)
    }
}

// MARK: - Live TV View
struct LiveTVView: View {
    @StateObject private var viewModel = LiveTVViewModel()
    @State private var selectedCategory = "All"
    
    var body: some View {
        NavigationView {
            VStack {
                // Category Filter
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        ForEach(viewModel.categories, id: \.self) { category in
                            CategoryChip(
                                title: category,
                                isSelected: selectedCategory == category
                            ) {
                                selectedCategory = category
                                viewModel.filterChannels(by: category)
                            }
                        }
                    }
                    .padding(.horizontal)
                }
                
                // Channels Grid
                ScrollView {
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 16) {
                        ForEach(viewModel.filteredChannels) { channel in
                            ChannelCard(channel: channel) {
                                viewModel.selectChannel(channel)
                            }
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle("Live TV")
            .onAppear {
                viewModel.loadChannels()
            }
            .sheet(item: $viewModel.selectedChannel) { channel in
                VideoPlayerView(channel: channel)
            }
        }
    }
}

// MARK: - Channel Card Component
struct ChannelCard: View {
    let channel: Channel
    let onTap: () -> Void
    
    var body: some View {
        VStack {
            AsyncImage(url: URL(string: channel.logoURL)) { image in
                image
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            } placeholder: {
                Rectangle()
                    .fill(Color.gray.opacity(0.3))
                    .overlay(
                        Image(systemName: "tv")
                            .foregroundColor(.gray)
                    )
            }
            .frame(height: 80)
            .cornerRadius(8)
            
            Text(channel.name)
                .font(.caption)
                .fontWeight(.medium)
                .multilineTextAlignment(.center)
            
            Text(channel.category)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(12)
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
        .onTapGesture {
            onTap()
        }
    }
}

// MARK: - Video Player View
struct VideoPlayerView: View {
    let channel: Channel
    @StateObject private var playerManager = VideoPlayerManager()
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        NavigationView {
            VStack {
                // Video Player
                VideoPlayer(player: playerManager.player)
                    .frame(height: 250)
                    .onAppear {
                        playerManager.setupPlayer(for: channel)
                        playerManager.play()
                    }
                    .onDisappear {
                        playerManager.pause()
                    }
                
                // Channel Info
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        AsyncImage(url: URL(string: channel.logoURL)) { image in
                            image
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                        } placeholder: {
                            Rectangle()
                                .fill(Color.gray.opacity(0.3))
                        }
                        .frame(width: 60, height: 40)
                        .cornerRadius(6)
                        
                        VStack(alignment: .leading) {
                            Text(channel.name)
                                .font(.headline)
                            Text(channel.description)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Button(action: {
                            // Toggle favorite
                        }) {
                            Image(systemName: "heart")
                                .foregroundColor(.streamVaultBlue)
                        }
                    }
                    
                    // Now Playing
                    if let currentProgram = channel.currentProgram {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Now Playing")
                                .font(.caption)
                                .fontWeight(.semibold)
                                .foregroundColor(.streamVaultBlue)
                            
                            Text(currentProgram.title)
                                .font(.body)
                                .fontWeight(.medium)
                            
                            Text(currentProgram.description)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                    }
                }
                .padding()
                
                Spacer()
            }
            .navigationTitle(channel.name)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Authentication Manager
class AuthenticationManager: ObservableObject {
    @Published var isAuthenticated = false
    @Published var currentUser: User?
    @Published var isLoading = false
    
    private let apiService = APIService.shared
    private var cancellables = Set<AnyCancellable>()
    
    func checkAuthStatus() {
        if let token = UserDefaults.standard.string(forKey: "auth_token") {
            apiService.setAuthToken(token)
            validateToken()
        }
    }
    
    func login(email: String, password: String) {
        isLoading = true
        
        apiService.login(email: email, password: password)
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { [weak self] completion in
                    self?.isLoading = false
                    if case .failure(let error) = completion {
                        print("Login error: \(error)")
                    }
                },
                receiveValue: { [weak self] response in
                    self?.handleAuthSuccess(response)
                }
            )
            .store(in: &cancellables)
    }
    
    func register(email: String, username: String, password: String) {
        isLoading = true
        
        apiService.register(email: email, username: username, password: password)
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { [weak self] completion in
                    self?.isLoading = false
                    if case .failure(let error) = completion {
                        print("Registration error: \(error)")
                    }
                },
                receiveValue: { [weak self] response in
                    self?.handleAuthSuccess(response)
                }
            )
            .store(in: &cancellables)
    }
    
    private func handleAuthSuccess(_ response: AuthResponse) {
        UserDefaults.standard.set(response.token, forKey: "auth_token")
        apiService.setAuthToken(response.token)
        currentUser = response.user
        isAuthenticated = true
    }
    
    private func validateToken() {
        apiService.validateToken()
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { completion in
                    if case .failure = completion {
                        self.logout()
                    }
                },
                receiveValue: { [weak self] user in
                    self?.currentUser = user
                    self?.isAuthenticated = true
                }
            )
            .store(in: &cancellables)
    }
    
    func logout() {
        UserDefaults.standard.removeObject(forKey: "auth_token")
        apiService.setAuthToken(nil)
        currentUser = nil
        isAuthenticated = false
    }
    
    func refreshTokenIfNeeded() {
        // Implement token refresh logic
    }
}

// MARK: - Streaming Manager
class StreamingManager: ObservableObject {
    @Published var isConnected = false
    @Published var connectionQuality: ConnectionQuality = .good
    
    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "NetworkMonitor")
    
    func configure() {
        setupNetworkMonitoring()
    }
    
    private func setupNetworkMonitoring() {
        monitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                self?.isConnected = path.status == .satisfied
                self?.updateConnectionQuality(path)
            }
        }
        monitor.start(queue: queue)
    }
    
    private func updateConnectionQuality(_ path: NWPath) {
        if path.usesInterfaceType(.wifi) {
            connectionQuality = .excellent
        } else if path.usesInterfaceType(.cellular) {
            connectionQuality = .good
        } else {
            connectionQuality = .poor
        }
    }
}

// MARK: - API Service
class APIService {
    static let shared = APIService()
    
    private let baseURL = "https://api.streamvault.pro"
    private var authToken: String?
    private let session = URLSession.shared
    
    func setAuthToken(_ token: String?) {
        authToken = token
    }
    
    func login(email: String, password: String) -> AnyPublisher<AuthResponse, Error> {
        let loginData = LoginRequest(email: email, password: password)
        return post("/auth/login", body: loginData)
    }
    
    func register(email: String, username: String, password: String) -> AnyPublisher<AuthResponse, Error> {
        let registerData = RegisterRequest(email: email, username: username, password: password)
        return post("/auth/register", body: registerData)
    }
    
    func validateToken() -> AnyPublisher<User, Error> {
        return get("/user/profile")
    }
    
    func getChannels(category: String? = nil) -> AnyPublisher<[Channel], Error> {
        var endpoint = "/content/channels"
        if let category = category {
            endpoint += "?category=\(category)"
        }
        return get(endpoint)
    }
    
    func getStreamingURL(for channelId: String) -> AnyPublisher<StreamingResponse, Error> {
        let streamData = StreamingRequest(contentId: channelId, contentType: "channel", deviceId: UIDevice.current.identifierForVendor?.uuidString ?? "unknown")
        return post("/streaming/authorize", body: streamData)
    }
    
    private func get<T: Codable>(_ endpoint: String) -> AnyPublisher<T, Error> {
        guard let url = URL(string: baseURL + endpoint) else {
            return Fail(error: APIError.invalidURL)
                .eraseToAnyPublisher()
        }
        
        var request = URLRequest(url: url)
        if let token = authToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        
        return session.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: T.self, decoder: JSONDecoder())
            .eraseToAnyPublisher()
    }
    
    private func post<T: Codable, U: Codable>(_ endpoint: String, body: U) -> AnyPublisher<T, Error> {
        guard let url = URL(string: baseURL + endpoint) else {
            return Fail(error: APIError.invalidURL)
                .eraseToAnyPublisher()
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        if let token = authToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        
        do {
            request.httpBody = try JSONEncoder().encode(body)
        } catch {
            return Fail(error: error)
                .eraseToAnyPublisher()
        }
        
        return session.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: T.self, decoder: JSONDecoder())
            .eraseToAnyPublisher()
    }
}

// MARK: - Data Models
struct User: Codable {
    let id: String
    let email: String
    let username: String
    let firstName: String?
    let lastName: String?
    let subscriptionTier: String
    let profileImageUrl: String?
}

struct Channel: Codable, Identifiable {
    let id: String
    let name: String
    let description: String
    let logoURL: String
    let category: String
    let isLive: Bool
    let currentProgram: Program?
}

struct Program: Codable {
    let title: String
    let description: String
    let startTime: Date
    let endTime: Date
}

struct AuthResponse: Codable {
    let user: User
    let token: String
    let message: String
}

struct LoginRequest: Codable {
    let email: String
    let password: String
}

struct RegisterRequest: Codable {
    let email: String
    let username: String
    let password: String
}

struct StreamingRequest: Codable {
    let contentId: String
    let contentType: String
    let deviceId: String
}

struct StreamingResponse: Codable {
    let sessionId: String
    let streamToken: String
    let streamingUrls: [String: String]
    let expiresAt: Date
}

enum APIError: Error {
    case invalidURL
    case noData
    case decodingError
}

enum ConnectionQuality {
    case poor, good, excellent
}

// MARK: - Extensions
extension Color {
    static let streamVaultBlue = Color(red: 0.12, green: 0.25, blue: 0.69)
    static let streamVaultAccent = Color(red: 0.02, green: 0.71, blue: 0.83)
}

// MARK: - Additional Views (Placeholder implementations)
struct AuthenticationView: View {
    var body: some View {
        Text("Authentication View")
    }
}

struct MoviesView: View {
    var body: some View {
        Text("Movies View")
    }
}

struct SportsView: View {
    var body: some View {
        Text("Sports View")
    }
}

struct MyLibraryView: View {
    var body: some View {
        Text("My Library View")
    }
}

struct ProfileView: View {
    var body: some View {
        Text("Profile View")
    }
}

struct CategoryChip: View {
    let title: String
    let isSelected: Bool
    let onTap: () -> Void
    
    var body: some View {
        Button(action: onTap) {
            Text(title)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(isSelected ? Color.streamVaultBlue : Color.gray.opacity(0.2))
                .foregroundColor(isSelected ? .white : .primary)
                .cornerRadius(20)
        }
    }
}

// MARK: - ViewModels
class LiveTVViewModel: ObservableObject {
    @Published var channels: [Channel] = []
    @Published var filteredChannels: [Channel] = []
    @Published var selectedChannel: Channel?
    @Published var categories = ["All", "News", "Sports", "Entertainment", "Movies", "Kids"]
    
    private let apiService = APIService.shared
    private var cancellables = Set<AnyCancellable>()
    
    func loadChannels() {
        apiService.getChannels()
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { completion in
                    if case .failure(let error) = completion {
                        print("Error loading channels: \(error)")
                    }
                },
                receiveValue: { [weak self] channels in
                    self?.channels = channels
                    self?.filteredChannels = channels
                }
            )
            .store(in: &cancellables)
    }
    
    func filterChannels(by category: String) {
        if category == "All" {
            filteredChannels = channels
        } else {
            filteredChannels = channels.filter { $0.category == category }
        }
    }
    
    func selectChannel(_ channel: Channel) {
        selectedChannel = channel
    }
}

class VideoPlayerManager: ObservableObject {
    @Published var player: AVPlayer?
    @Published var isPlaying = false
    
    private let apiService = APIService.shared
    private var cancellables = Set<AnyCancellable>()
    
    func setupPlayer(for channel: Channel) {
        apiService.getStreamingURL(for: channel.id)
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { completion in
                    if case .failure(let error) = completion {
                        print("Error getting streaming URL: \(error)")
                    }
                },
                receiveValue: { [weak self] response in
                    if let urlString = response.streamingUrls["HD"],
                       let url = URL(string: urlString) {
                        self?.player = AVPlayer(url: url)
                    }
                }
            )
            .store(in: &cancellables)
    }
    
    func play() {
        player?.play()
        isPlaying = true
    }
    
    func pause() {
        player?.pause()
        isPlaying = false
    }
}

// MARK: - Analytics Manager
class AnalyticsManager {
    static let shared = AnalyticsManager()
    
    private init() {}
    
    func initialize() {
        // Initialize analytics SDK
        print("Analytics initialized")
    }
    
    func trackEvent(_ event: String, parameters: [String: Any] = [:]) {
        // Track analytics event
        print("Analytics event: \(event) with parameters: \(parameters)")
    }
}

// MARK: - Info.plist Configuration Required
/*
Add to Info.plist:

<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <true/>
</dict>

<key>NSCameraUsageDescription</key>
<string>StreamVault Pro needs camera access for profile pictures</string>

<key>NSMicrophoneUsageDescription</key>
<string>StreamVault Pro needs microphone access for voice commands</string>

<key>CFBundleDisplayName</key>
<string>StreamVault Pro</string>

<key>CFBundleIdentifier</key>
<string>com.streamvault.pro</string>

<key>LSRequiresIPhoneOS</key>
<true/>

<key>UILaunchStoryboardName</key>
<string>LaunchScreen</string>

<key>UISupportedInterfaceOrientations</key>
<array>
    <string>UIInterfaceOrientationPortrait</string>
    <string>UIInterfaceOrientationLandscapeLeft</string>
    <string>UIInterfaceOrientationLandscapeRight</string>
</array>
*/