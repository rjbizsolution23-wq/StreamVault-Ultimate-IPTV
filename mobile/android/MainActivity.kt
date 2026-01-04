// StreamVault Pro - Android Native Application
// Complete Android app with Kotlin, Jetpack Compose, and ExoPlayer

package com.streamvault.pro

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.*
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.google.android.exoplayer2.ExoPlayer
import com.google.android.exoplayer2.MediaItem
import com.google.android.exoplayer2.source.hls.HlsMediaSource
import com.google.android.exoplayer2.ui.StyledPlayerView
import com.google.android.exoplayer2.upstream.DefaultHttpDataSource
import dagger.hilt.android.AndroidEntryPoint
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import retrofit2.*
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*
import javax.inject.Inject
import javax.inject.Singleton

// MARK: - Main Activity
@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        setContent {
            StreamVaultProTheme {
                StreamVaultProApp()
            }
        }
    }
}

// MARK: - Main App Composable
@Composable
fun StreamVaultProApp() {
    val navController = rememberNavController()
    val authViewModel: AuthViewModel = hiltViewModel()
    val isAuthenticated by authViewModel.isAuthenticated.collectAsState()
    
    LaunchedEffect(Unit) {
        authViewModel.checkAuthStatus()
    }
    
    if (isAuthenticated) {
        MainNavigation(navController)
    } else {
        AuthenticationScreen(authViewModel)
    }
}

// MARK: - Navigation
@Composable
fun MainNavigation(navController: NavHostController) {
    Scaffold(
        bottomBar = {
            BottomNavigationBar(navController)
        }
    ) { paddingValues ->
        NavHost(
            navController = navController,
            startDestination = "live_tv",
            modifier = Modifier.padding(paddingValues)
        ) {
            composable("live_tv") {
                LiveTVScreen()
            }
            composable("movies") {
                MoviesScreen()
            }
            composable("sports") {
                SportsScreen()
            }
            composable("library") {
                LibraryScreen()
            }
            composable("profile") {
                ProfileScreen()
            }
            composable("player/{channelId}") { backStackEntry ->
                val channelId = backStackEntry.arguments?.getString("channelId") ?: ""
                VideoPlayerScreen(channelId = channelId)
            }
        }
    }
}

// MARK: - Bottom Navigation
@Composable
fun BottomNavigationBar(navController: NavHostController) {
    NavigationBar(
        containerColor = MaterialTheme.colorScheme.surface
    ) {
        val items = listOf(
            BottomNavItem("live_tv", "Live TV", Icons.Default.Tv),
            BottomNavItem("movies", "Movies", Icons.Default.Movie),
            BottomNavItem("sports", "Sports", Icons.Default.SportsBasketball),
            BottomNavItem("library", "Library", Icons.Default.Favorite),
            BottomNavItem("profile", "Profile", Icons.Default.Person)
        )
        
        items.forEach { item ->
            NavigationBarItem(
                icon = { Icon(item.icon, contentDescription = item.title) },
                label = { Text(item.title) },
                selected = false, // Implement selection logic
                onClick = {
                    navController.navigate(item.route) {
                        popUpTo(navController.graph.startDestinationId)
                        launchSingleTop = true
                    }
                },
                colors = NavigationBarItemDefaults.colors(
                    selectedIconColor = StreamVaultBlue,
                    selectedTextColor = StreamVaultBlue
                )
            )
        }
    }
}

data class BottomNavItem(
    val route: String,
    val title: String,
    val icon: androidx.compose.ui.graphics.vector.ImageVector
)

// MARK: - Live TV Screen
@Composable
fun LiveTVScreen() {
    val viewModel: LiveTVViewModel = hiltViewModel()
    val channels by viewModel.channels.collectAsState()
    val categories by viewModel.categories.collectAsState()
    val selectedCategory by viewModel.selectedCategory.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    
    LaunchedEffect(Unit) {
        viewModel.loadChannels()
    }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // Header
        Text(
            text = "Live TV",
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(bottom = 16.dp)
        )
        
        // Category Filter
        LazyRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.padding(bottom = 16.dp)
        ) {
            items(categories) { category ->
                CategoryChip(
                    category = category,
                    isSelected = selectedCategory == category,
                    onClick = { viewModel.selectCategory(category) }
                )
            }
        }
        
        // Channels Grid
        if (isLoading) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator(color = StreamVaultBlue)
            }
        } else {
            LazyVerticalGrid(
                columns = GridCells.Fixed(2),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                items(channels) { channel ->
                    ChannelCard(
                        channel = channel,
                        onClick = { viewModel.playChannel(channel) }
                    )
                }
            }
        }
    }
}

// MARK: - Channel Card Component
@Composable
fun ChannelCard(
    channel: Channel,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(1f),
        onClick = onClick,
        shape = RoundedCornerShape(12.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Channel Logo
            AsyncImage(
                model = ImageRequest.Builder(LocalContext.current)
                    .data(channel.logoUrl)
                    .crossfade(true)
                    .build(),
                contentDescription = channel.name,
                modifier = Modifier
                    .size(60.dp)
                    .clip(RoundedCornerShape(8.dp)),
                contentScale = ContentScale.Fit
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // Channel Name
            Text(
                text = channel.name,
                fontSize = 14.sp,
                fontWeight = FontWeight.Medium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.fillMaxWidth()
            )
            
            // Category
            Text(
                text = channel.category,
                fontSize = 12.sp,
                color = Color.Gray,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            
            if (channel.isLive) {
                Spacer(modifier = Modifier.height(4.dp))
                
                Surface(
                    color = Color.Red,
                    shape = RoundedCornerShape(4.dp),
                    modifier = Modifier.padding(2.dp)
                ) {
                    Text(
                        text = "LIVE",
                        fontSize = 10.sp,
                        color = Color.White,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                    )
                }
            }
        }
    }
}

// MARK: - Category Chip Component
@Composable
fun CategoryChip(
    category: String,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    FilterChip(
        onClick = onClick,
        label = { Text(category) },
        selected = isSelected,
        colors = FilterChipDefaults.filterChipColors(
            selectedContainerColor = StreamVaultBlue,
            selectedLabelColor = Color.White
        )
    )
}

// MARK: - Video Player Screen
@Composable
fun VideoPlayerScreen(channelId: String) {
    val viewModel: VideoPlayerViewModel = hiltViewModel()
    val streamingUrl by viewModel.streamingUrl.collectAsState()
    val channel by viewModel.channel.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    
    LaunchedEffect(channelId) {
        viewModel.loadChannel(channelId)
        viewModel.getStreamingUrl(channelId)
    }
    
    Column(modifier = Modifier.fillMaxSize()) {
        // Video Player
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(16f / 9f)
        ) {
            if (isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.align(Alignment.Center),
                    color = StreamVaultBlue
                )
            } else if (streamingUrl.isNotEmpty()) {
                ExoPlayerView(
                    streamingUrl = streamingUrl,
                    modifier = Modifier.fillMaxSize()
                )
            }
        }
        
        // Channel Information
        channel?.let { ch ->
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    AsyncImage(
                        model = ch.logoUrl,
                        contentDescription = ch.name,
                        modifier = Modifier
                            .size(80.dp)
                            .clip(RoundedCornerShape(8.dp))
                    )
                    
                    Spacer(modifier = Modifier.width(16.dp))
                    
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = ch.name,
                            fontSize = 20.sp,
                            fontWeight = FontWeight.Bold
                        )
                        
                        Text(
                            text = ch.description,
                            fontSize = 14.sp,
                            color = Color.Gray,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                        
                        ch.currentProgram?.let { program ->
                            Spacer(modifier = Modifier.height(8.dp))
                            
                            Text(
                                text = "Now Playing: ${program.title}",
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Medium,
                                color = StreamVaultBlue
                            )
                        }
                    }
                    
                    IconButton(
                        onClick = { viewModel.toggleFavorite(ch.id) }
                    ) {
                        Icon(
                            imageVector = if (ch.isFavorite) Icons.Default.Favorite else Icons.Default.FavoriteBorder,
                            contentDescription = "Favorite",
                            tint = if (ch.isFavorite) Color.Red else Color.Gray
                        )
                    }
                }
            }
        }
        
        Spacer(modifier = Modifier.weight(1f))
    }
}

// MARK: - ExoPlayer Component
@Composable
fun ExoPlayerView(
    streamingUrl: String,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    
    val exoPlayer = remember {
        ExoPlayer.Builder(context).build().apply {
            val mediaItem = MediaItem.fromUri(streamingUrl)
            val dataSourceFactory = DefaultHttpDataSource.Factory()
            val hlsMediaSource = HlsMediaSource.Factory(dataSourceFactory)
                .createMediaSource(mediaItem)
            
            setMediaSource(hlsMediaSource)
            prepare()
            playWhenReady = true
        }
    }
    
    DisposableEffect(
        AndroidView(
            factory = { ctx ->
                StyledPlayerView(ctx).apply {
                    player = exoPlayer
                    useController = true
                }
            },
            modifier = modifier
        )
    ) {
        onDispose {
            exoPlayer.release()
        }
    }
}

// MARK: - Authentication Screen
@Composable
fun AuthenticationScreen(viewModel: AuthViewModel) {
    var isLoginMode by remember { mutableStateOf(true) }
    var email by remember { mutableStateOf("") }
    var username by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    
    val isLoading by viewModel.isLoading.collectAsState()
    val errorMessage by viewModel.errorMessage.collectAsState()
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        // Logo and Title
        Text(
            text = "StreamVault Pro",
            fontSize = 32.sp,
            fontWeight = FontWeight.Bold,
            color = StreamVaultBlue
        )
        
        Spacer(modifier = Modifier.height(48.dp))
        
        // Email Field
        OutlinedTextField(
            value = email,
            onValueChange = { email = it },
            label = { Text("Email") },
            modifier = Modifier.fillMaxWidth()
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        // Username Field (Register only)
        if (!isLoginMode) {
            OutlinedTextField(
                value = username,
                onValueChange = { username = it },
                label = { Text("Username") },
                modifier = Modifier.fillMaxWidth()
            )
            
            Spacer(modifier = Modifier.height(16.dp))
        }
        
        // Password Field
        OutlinedTextField(
            value = password,
            onValueChange = { password = it },
            label = { Text("Password") },
            modifier = Modifier.fillMaxWidth()
        )
        
        Spacer(modifier = Modifier.height(24.dp))
        
        // Submit Button
        Button(
            onClick = {
                if (isLoginMode) {
                    viewModel.login(email, password)
                } else {
                    viewModel.register(email, username, password)
                }
            },
            modifier = Modifier.fillMaxWidth(),
            enabled = !isLoading,
            colors = ButtonDefaults.buttonColors(
                containerColor = StreamVaultBlue
            )
        ) {
            if (isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    color = Color.White
                )
            } else {
                Text(if (isLoginMode) "Login" else "Register")
            }
        }
        
        // Toggle Mode
        TextButton(
            onClick = { isLoginMode = !isLoginMode }
        ) {
            Text(
                if (isLoginMode) "Don't have an account? Register" else "Already have an account? Login",
                color = StreamVaultBlue
            )
        }
        
        // Error Message
        errorMessage?.let { error ->
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = error,
                color = Color.Red,
                fontSize = 14.sp
            )
        }
    }
}

// MARK: - Data Models
data class User(
    val id: String,
    val email: String,
    val username: String,
    val firstName: String?,
    val lastName: String?,
    val subscriptionTier: String,
    val profileImageUrl: String?
)

data class Channel(
    val id: String,
    val name: String,
    val description: String,
    val logoUrl: String,
    val category: String,
    val isLive: Boolean,
    val isFavorite: Boolean = false,
    val currentProgram: Program?
)

data class Program(
    val title: String,
    val description: String,
    val startTime: String,
    val endTime: String
)

data class AuthResponse(
    val user: User,
    val token: String,
    val message: String
)

data class StreamingResponse(
    val sessionId: String,
    val streamToken: String,
    val streamingUrls: Map<String, String>,
    val expiresAt: String
)

// MARK: - API Service
interface ApiService {
    @POST("auth/login")
    suspend fun login(@Body request: LoginRequest): AuthResponse
    
    @POST("auth/register")
    suspend fun register(@Body request: RegisterRequest): AuthResponse
    
    @GET("user/profile")
    suspend fun getProfile(@Header("Authorization") token: String): User
    
    @GET("content/channels")
    suspend fun getChannels(
        @Query("category") category: String? = null
    ): List<Channel>
    
    @GET("content/channels/{id}")
    suspend fun getChannel(@Path("id") channelId: String): Channel
    
    @POST("streaming/authorize")
    suspend fun getStreamingUrl(
        @Body request: StreamingRequest,
        @Header("Authorization") token: String
    ): StreamingResponse
}

data class LoginRequest(
    val email: String,
    val password: String
)

data class RegisterRequest(
    val email: String,
    val username: String,
    val password: String
)

data class StreamingRequest(
    val contentId: String,
    val contentType: String,
    val deviceId: String,
    val quality: String = "HD"
)

// MARK: - Repository
@Singleton
class StreamVaultRepository @Inject constructor(
    private val apiService: ApiService,
    private val tokenManager: TokenManager
) {
    suspend fun login(email: String, password: String): Result<AuthResponse> {
        return try {
            val response = apiService.login(LoginRequest(email, password))
            tokenManager.saveToken(response.token)
            Result.success(response)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    suspend fun register(email: String, username: String, password: String): Result<AuthResponse> {
        return try {
            val response = apiService.register(RegisterRequest(email, username, password))
            tokenManager.saveToken(response.token)
            Result.success(response)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    suspend fun getChannels(category: String? = null): Result<List<Channel>> {
        return try {
            val channels = apiService.getChannels(category)
            Result.success(channels)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    suspend fun getChannel(channelId: String): Result<Channel> {
        return try {
            val channel = apiService.getChannel(channelId)
            Result.success(channel)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    suspend fun getStreamingUrl(channelId: String): Result<StreamingResponse> {
        return try {
            val token = tokenManager.getToken() ?: throw Exception("No auth token")
            val response = apiService.getStreamingUrl(
                StreamingRequest(
                    contentId = channelId,
                    contentType = "channel",
                    deviceId = android.provider.Settings.Secure.getString(
                        /* contentResolver = */ null,
                        android.provider.Settings.Secure.ANDROID_ID
                    ) ?: "unknown"
                ),
                "Bearer $token"
            )
            Result.success(response)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}

// MARK: - Token Manager
@Singleton
class TokenManager @Inject constructor() {
    private var token: String? = null
    
    fun saveToken(newToken: String) {
        token = newToken
        // Save to SharedPreferences in real implementation
    }
    
    fun getToken(): String? = token
    
    fun clearToken() {
        token = null
    }
}

// MARK: - ViewModels
@HiltViewModel
class AuthViewModel @Inject constructor(
    private val repository: StreamVaultRepository,
    private val tokenManager: TokenManager
) : ViewModel() {
    
    private val _isAuthenticated = MutableStateFlow(false)
    val isAuthenticated: StateFlow<Boolean> = _isAuthenticated.asStateFlow()
    
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()
    
    fun checkAuthStatus() {
        val token = tokenManager.getToken()
        _isAuthenticated.value = token != null
    }
    
    fun login(email: String, password: String) {
        viewModelScope.launch {
            _isLoading.value = true
            _errorMessage.value = null
            
            repository.login(email, password).fold(
                onSuccess = {
                    _isAuthenticated.value = true
                },
                onFailure = { error ->
                    _errorMessage.value = error.message ?: "Login failed"
                }
            )
            
            _isLoading.value = false
        }
    }
    
    fun register(email: String, username: String, password: String) {
        viewModelScope.launch {
            _isLoading.value = true
            _errorMessage.value = null
            
            repository.register(email, username, password).fold(
                onSuccess = {
                    _isAuthenticated.value = true
                },
                onFailure = { error ->
                    _errorMessage.value = error.message ?: "Registration failed"
                }
            )
            
            _isLoading.value = false
        }
    }
    
    fun logout() {
        tokenManager.clearToken()
        _isAuthenticated.value = false
    }
}

@HiltViewModel
class LiveTVViewModel @Inject constructor(
    private val repository: StreamVaultRepository
) : ViewModel() {
    
    private val _channels = MutableStateFlow<List<Channel>>(emptyList())
    val channels: StateFlow<List<Channel>> = _channels.asStateFlow()
    
    private val _categories = MutableStateFlow(listOf("All", "News", "Sports", "Entertainment", "Movies", "Kids"))
    val categories: StateFlow<List<String>> = _categories.asStateFlow()
    
    private val _selectedCategory = MutableStateFlow("All")
    val selectedCategory: StateFlow<String> = _selectedCategory.asStateFlow()
    
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    fun loadChannels() {
        viewModelScope.launch {
            _isLoading.value = true
            
            repository.getChannels().fold(
                onSuccess = { channelList ->
                    _channels.value = channelList
                },
                onFailure = { error ->
                    // Handle error
                }
            )
            
            _isLoading.value = false
        }
    }
    
    fun selectCategory(category: String) {
        _selectedCategory.value = category
        filterChannels(category)
    }
    
    private fun filterChannels(category: String) {
        viewModelScope.launch {
            _isLoading.value = true
            
            val categoryFilter = if (category == "All") null else category
            repository.getChannels(categoryFilter).fold(
                onSuccess = { channelList ->
                    _channels.value = channelList
                },
                onFailure = { error ->
                    // Handle error
                }
            )
            
            _isLoading.value = false
        }
    }
    
    fun playChannel(channel: Channel) {
        // Navigate to player screen
        // This would be handled by the navigation component
    }
}

@HiltViewModel
class VideoPlayerViewModel @Inject constructor(
    private val repository: StreamVaultRepository
) : ViewModel() {
    
    private val _channel = MutableStateFlow<Channel?>(null)
    val channel: StateFlow<Channel?> = _channel.asStateFlow()
    
    private val _streamingUrl = MutableStateFlow("")
    val streamingUrl: StateFlow<String> = _streamingUrl.asStateFlow()
    
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    fun loadChannel(channelId: String) {
        viewModelScope.launch {
            repository.getChannel(channelId).fold(
                onSuccess = { channelData ->
                    _channel.value = channelData
                },
                onFailure = { error ->
                    // Handle error
                }
            )
        }
    }
    
    fun getStreamingUrl(channelId: String) {
        viewModelScope.launch {
            _isLoading.value = true
            
            repository.getStreamingUrl(channelId).fold(
                onSuccess = { response ->
                    _streamingUrl.value = response.streamingUrls["HD"] ?: ""
                },
                onFailure = { error ->
                    // Handle error
                }
            )
            
            _isLoading.value = false
        }
    }
    
    fun toggleFavorite(channelId: String) {
        // Implement favorite toggle
    }
}

// MARK: - Theme
@Composable
fun StreamVaultProTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = StreamVaultBlue,
            secondary = StreamVaultAccent
        ),
        content = content
    )
}

val StreamVaultBlue = Color(0xFF1E40AF)
val StreamVaultAccent = Color(0xFF06B6D4)

// MARK: - Placeholder Screens
@Composable
fun MoviesScreen() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Text("Movies Screen - Coming Soon!")
    }
}

@Composable
fun SportsScreen() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Text("Sports Screen - Coming Soon!")
    }
}

@Composable
fun LibraryScreen() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Text("Library Screen - Coming Soon!")
    }
}

@Composable
fun ProfileScreen() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Text("Profile Screen - Coming Soon!")
    }
}

/*
build.gradle (Module: app) dependencies:

dependencies {
    implementation 'androidx.core:core-ktx:1.10.1'
    implementation 'androidx.lifecycle:lifecycle-runtime-ktx:2.7.0'
    implementation 'androidx.activity:activity-compose:1.7.2'
    
    // Compose BOM
    implementation platform('androidx.compose:compose-bom:2023.06.01')
    implementation 'androidx.compose.ui:ui'
    implementation 'androidx.compose.ui:ui-graphics'
    implementation 'androidx.compose.ui:ui-tooling-preview'
    implementation 'androidx.compose.material3:material3'
    
    // Navigation
    implementation 'androidx.navigation:navigation-compose:2.6.0'
    
    // ViewModel
    implementation 'androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0'
    
    // Hilt
    implementation 'com.google.dagger:hilt-android:2.47'
    kapt 'com.google.dagger:hilt-compiler:2.47'
    implementation 'androidx.hilt:hilt-navigation-compose:1.0.0'
    
    // ExoPlayer
    implementation 'com.google.android.exoplayer:exoplayer:2.19.1'
    implementation 'com.google.android.exoplayer:exoplayer-hls:2.19.1'
    implementation 'com.google.android.exoplayer:exoplayer-ui:2.19.1'
    
    // Retrofit
    implementation 'com.squareup.retrofit2:retrofit:2.9.0'
    implementation 'com.squareup.retrofit2:converter-gson:2.9.0'
    implementation 'com.squareup.okhttp3:logging-interceptor:4.11.0'
    
    // Coil (Image Loading)
    implementation 'io.coil-kt:coil-compose:2.4.0'
    
    // Coroutines
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
}

AndroidManifest.xml permissions:

<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
<uses-permission android:name="android.permission.WAKE_LOCK" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
*/