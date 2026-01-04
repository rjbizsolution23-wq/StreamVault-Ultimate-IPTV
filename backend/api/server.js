// StreamVault Pro - Main API Server
// Production-ready Express.js backend with all IPTV features

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const { Pool } = require('pg');
const Redis = require('redis');
const WebSocket = require('ws');
const { createServer } = require('http');
const { v4: uuidv4 } = require('uuid');
const multer = require('multer');
const sharp = require('sharp');
const ffmpeg = require('fluent-ffmpeg');

const app = express();
const server = createServer(app);
const wss = new WebSocket.Server({ server });

// ==========================================
// CONFIGURATION
// ==========================================

const config = {
    port: process.env.PORT || 3000,
    jwtSecret: process.env.JWT_SECRET || 'streamvault_super_secret_key_2024',
    database: {
        host: process.env.DB_HOST || 'localhost',
        port: process.env.DB_PORT || 5432,
        database: process.env.DB_NAME || 'streamvault_pro',
        user: process.env.DB_USER || 'postgres',
        password: process.env.DB_PASS || 'password'
    },
    redis: {
        host: process.env.REDIS_HOST || 'localhost',
        port: process.env.REDIS_PORT || 6379
    },
    cdn: {
        baseUrl: process.env.CDN_BASE_URL || 'https://cdn.streamvault.pro',
        streamingUrl: process.env.STREAMING_URL || 'https://stream.streamvault.pro'
    },
    stripe: {
        secretKey: process.env.STRIPE_SECRET_KEY,
        webhookSecret: process.env.STRIPE_WEBHOOK_SECRET
    }
};

// ==========================================
// DATABASE & REDIS CONNECTIONS
// ==========================================

const db = new Pool(config.database);
const redis = Redis.createClient(config.redis);

redis.on('error', (err) => console.error('Redis Client Error', err));
redis.connect();

// ==========================================
// MIDDLEWARE SETUP
// ==========================================

// Security & Performance
app.use(helmet());
app.use(compression());
app.use(cors({
    origin: ['http://localhost:3000', 'https://streamvault.pro', 'https://www.streamvault.pro'],
    credentials: true
}));

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP to 100 requests per windowMs
    message: 'Too many requests from this IP'
});
app.use('/api/', limiter);

// Stricter rate limiting for auth endpoints
const authLimiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 5, // limit each IP to 5 requests per windowMs
    message: 'Too many authentication attempts'
});
app.use('/api/auth/', authLimiter);

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// File upload configuration
const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
    fileFilter: (req, file, cb) => {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'video/mp4'];
        cb(null, allowedTypes.includes(file.mimetype));
    }
});

// ==========================================
// AUTHENTICATION MIDDLEWARE
// ==========================================

const authenticateToken = async (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
        return res.status(401).json({ error: 'Access token required' });
    }

    try {
        const decoded = jwt.verify(token, config.jwtSecret);
        const user = await db.query('SELECT * FROM users WHERE id = $1 AND is_active = true', [decoded.userId]);
        
        if (user.rows.length === 0) {
            return res.status(401).json({ error: 'User not found or inactive' });
        }

        req.user = user.rows[0];
        next();
    } catch (error) {
        return res.status(403).json({ error: 'Invalid token' });
    }
};

// Admin authentication middleware
const authenticateAdmin = async (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
        return res.status(401).json({ error: 'Admin access token required' });
    }

    try {
        const decoded = jwt.verify(token, config.jwtSecret);
        const admin = await db.query('SELECT * FROM admin_users WHERE id = $1 AND is_active = true', [decoded.adminId]);
        
        if (admin.rows.length === 0) {
            return res.status(401).json({ error: 'Admin not found or inactive' });
        }

        req.admin = admin.rows[0];
        next();
    } catch (error) {
        return res.status(403).json({ error: 'Invalid admin token' });
    }
};

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

const generateToken = (payload, expiresIn = '24h') => {
    return jwt.sign(payload, config.jwtSecret, { expiresIn });
};

const hashPassword = async (password) => {
    return await bcrypt.hash(password, 12);
};

const comparePassword = async (password, hash) => {
    return await bcrypt.compare(password, hash);
};

const generateStreamToken = (userId, contentId, deviceId) => {
    const payload = {
        userId,
        contentId,
        deviceId,
        timestamp: Date.now(),
        type: 'stream'
    };
    return jwt.sign(payload, config.jwtSecret, { expiresIn: '6h' });
};

// ==========================================
// AUTHENTICATION ROUTES
// ==========================================

// User registration
app.post('/api/auth/register', async (req, res) => {
    try {
        const { email, username, password, firstName, lastName, countryCode } = req.body;

        // Validation
        if (!email || !username || !password) {
            return res.status(400).json({ error: 'Email, username, and password required' });
        }

        // Check if user already exists
        const existingUser = await db.query(
            'SELECT id FROM users WHERE email = $1 OR username = $2',
            [email, username]
        );

        if (existingUser.rows.length > 0) {
            return res.status(400).json({ error: 'Email or username already exists' });
        }

        // Hash password
        const passwordHash = await hashPassword(password);

        // Create user
        const newUser = await db.query(
            `INSERT INTO users (email, username, password_hash, first_name, last_name, country_code)
             VALUES ($1, $2, $3, $4, $5, $6) RETURNING id, email, username, created_at`,
            [email, username, passwordHash, firstName, lastName, countryCode]
        );

        // Generate token
        const token = generateToken({ userId: newUser.rows[0].id });

        res.status(201).json({
            message: 'User created successfully',
            user: newUser.rows[0],
            token
        });
    } catch (error) {
        console.error('Registration error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// User login
app.post('/api/auth/login', async (req, res) => {
    try {
        const { emailOrUsername, password, deviceInfo } = req.body;

        if (!emailOrUsername || !password) {
            return res.status(400).json({ error: 'Email/username and password required' });
        }

        // Find user
        const user = await db.query(
            'SELECT * FROM users WHERE (email = $1 OR username = $1) AND is_active = true',
            [emailOrUsername]
        );

        if (user.rows.length === 0) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        const userData = user.rows[0];

        // Check password
        const isValidPassword = await comparePassword(password, userData.password_hash);
        if (!isValidPassword) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        // Update last login
        await db.query('UPDATE users SET last_login_at = NOW() WHERE id = $1', [userData.id]);

        // Create session
        const sessionToken = generateToken({ userId: userData.id, sessionId: uuidv4() }, '30d');
        
        await db.query(
            `INSERT INTO user_sessions (user_id, session_token, device_info, ip_address, user_agent, expires_at)
             VALUES ($1, $2, $3, $4, $5, NOW() + INTERVAL '30 days')`,
            [userData.id, sessionToken, deviceInfo, req.ip, req.get('User-Agent')]
        );

        // Generate API token
        const token = generateToken({ userId: userData.id });

        res.json({
            message: 'Login successful',
            user: {
                id: userData.id,
                email: userData.email,
                username: userData.username,
                firstName: userData.first_name,
                lastName: userData.last_name,
                subscriptionTier: userData.subscription_tier,
                profileImageUrl: userData.profile_image_url
            },
            token,
            sessionToken
        });
    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Token refresh
app.post('/api/auth/refresh', authenticateToken, async (req, res) => {
    try {
        const newToken = generateToken({ userId: req.user.id });
        res.json({ token: newToken });
    } catch (error) {
        console.error('Token refresh error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ==========================================
// USER PROFILE ROUTES
// ==========================================

// Get user profile
app.get('/api/user/profile', authenticateToken, async (req, res) => {
    try {
        const userProfile = await db.query(
            `SELECT u.*, us.status as subscription_status, sp.name as plan_name
             FROM users u
             LEFT JOIN user_subscriptions us ON u.id = us.user_id AND us.status = 'active'
             LEFT JOIN subscription_plans sp ON us.plan_id = sp.id
             WHERE u.id = $1`,
            [req.user.id]
        );

        if (userProfile.rows.length === 0) {
            return res.status(404).json({ error: 'User not found' });
        }

        const profile = userProfile.rows[0];
        delete profile.password_hash; // Remove sensitive data

        res.json({ profile });
    } catch (error) {
        console.error('Profile fetch error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Update user profile
app.put('/api/user/profile', authenticateToken, async (req, res) => {
    try {
        const { firstName, lastName, phone, languagePreference } = req.body;

        const updatedUser = await db.query(
            `UPDATE users 
             SET first_name = $1, last_name = $2, phone = $3, language_preference = $4, updated_at = NOW()
             WHERE id = $5 
             RETURNING id, email, username, first_name, last_name, phone, language_preference`,
            [firstName, lastName, phone, languagePreference, req.user.id]
        );

        res.json({
            message: 'Profile updated successfully',
            user: updatedUser.rows[0]
        });
    } catch (error) {
        console.error('Profile update error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ==========================================
// CONTENT DISCOVERY ROUTES
// ==========================================

// Get live TV channels
app.get('/api/content/channels', authenticateToken, async (req, res) => {
    try {
        const { category, page = 1, limit = 50, search } = req.query;
        const offset = (page - 1) * limit;

        let query = `
            SELECT lc.*, cc.name as category_name, cp.name as provider_name
            FROM live_channels lc
            LEFT JOIN content_categories cc ON lc.category_id = cc.id
            LEFT JOIN content_providers cp ON lc.provider_id = cp.id
            WHERE lc.is_active = true
        `;
        const params = [];

        if (category) {
            query += ' AND cc.slug = $' + (params.length + 1);
            params.push(category);
        }

        if (search) {
            query += ' AND (lc.name ILIKE $' + (params.length + 1) + ' OR lc.description ILIKE $' + (params.length + 1) + ')';
            params.push(`%${search}%`);
        }

        query += ' ORDER BY lc.sort_order, lc.channel_number LIMIT $' + (params.length + 1) + ' OFFSET $' + (params.length + 2);
        params.push(limit, offset);

        const channels = await db.query(query, params);

        // Get total count for pagination
        const countQuery = `
            SELECT COUNT(*) as total 
            FROM live_channels lc 
            LEFT JOIN content_categories cc ON lc.category_id = cc.id 
            WHERE lc.is_active = true
            ${category ? ' AND cc.slug = $1' : ''}
            ${search ? ` AND (lc.name ILIKE $${category ? 2 : 1} OR lc.description ILIKE $${category ? 2 : 1})` : ''}
        `;
        const countParams = category && search ? [category, `%${search}%`] : 
                           category ? [category] : 
                           search ? [`%${search}%`] : [];
        
        const totalCount = await db.query(countQuery, countParams);

        res.json({
            channels: channels.rows,
            pagination: {
                page: parseInt(page),
                limit: parseInt(limit),
                total: parseInt(totalCount.rows[0].total),
                totalPages: Math.ceil(totalCount.rows[0].total / limit)
            }
        });
    } catch (error) {
        console.error('Channels fetch error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Get VOD content (movies, series)
app.get('/api/content/vod', authenticateToken, async (req, res) => {
    try {
        const { category, type, page = 1, limit = 20, search, sortBy = 'created_at' } = req.query;
        const offset = (page - 1) * limit;

        let query = `
            SELECT vc.*, cc.name as category_name, cp.name as provider_name
            FROM vod_content vc
            LEFT JOIN content_categories cc ON vc.category_id = cc.id
            LEFT JOIN content_providers cp ON vc.provider_id = cp.id
            WHERE vc.is_active = true
        `;
        const params = [];

        if (category) {
            query += ' AND cc.slug = $' + (params.length + 1);
            params.push(category);
        }

        if (type) {
            query += ' AND vc.content_type = $' + (params.length + 1);
            params.push(type);
        }

        if (search) {
            query += ' AND (to_tsvector(\'english\', vc.title) @@ plainto_tsquery(\'english\', $' + (params.length + 1) + ') OR vc.description ILIKE $' + (params.length + 1) + ')';
            params.push(search, `%${search}%`);
        }

        // Sort options
        const sortOptions = {
            'created_at': 'vc.created_at DESC',
            'title': 'vc.title ASC',
            'rating': 'vc.rating_average DESC',
            'views': 'vc.view_count DESC',
            'release_date': 'vc.release_date DESC'
        };

        query += ' ORDER BY ' + (sortOptions[sortBy] || sortOptions['created_at']);
        query += ' LIMIT $' + (params.length + 1) + ' OFFSET $' + (params.length + 2);
        params.push(limit, offset);

        const content = await db.query(query, params);
        res.json({ content: content.rows });
    } catch (error) {
        console.error('VOD content fetch error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ==========================================
// STREAMING ROUTES
// ==========================================

// Generate streaming URL with authentication
app.post('/api/streaming/authorize', authenticateToken, async (req, res) => {
    try {
        const { contentId, contentType, deviceId, quality = 'HD' } = req.body;

        // Validate user subscription for content access
        const subscription = await db.query(
            'SELECT us.*, sp.* FROM user_subscriptions us JOIN subscription_plans sp ON us.plan_id = sp.id WHERE us.user_id = $1 AND us.status = \'active\'',
            [req.user.id]
        );

        if (subscription.rows.length === 0) {
            return res.status(403).json({ error: 'Active subscription required' });
        }

        const userPlan = subscription.rows[0];

        // Check concurrent streams limit
        const activeStreams = await db.query(
            'SELECT COUNT(*) as count FROM streaming_sessions WHERE user_id = $1 AND is_active = true',
            [req.user.id]
        );

        if (parseInt(activeStreams.rows[0].count) >= userPlan.max_concurrent_streams) {
            return res.status(429).json({ error: 'Maximum concurrent streams exceeded' });
        }

        // Generate secure streaming token
        const streamToken = generateStreamToken(req.user.id, contentId, deviceId);

        // Create streaming session
        const session = await db.query(
            `INSERT INTO streaming_sessions (user_id, device_id, content_type, content_id, session_token, quality)
             VALUES ($1, $2, $3, $4, $5, $6) RETURNING *`,
            [req.user.id, deviceId, contentType, contentId, streamToken, quality]
        );

        // Generate streaming URLs based on content type
        let streamingUrls = {};

        if (contentType === 'channel') {
            const channel = await db.query('SELECT stream_url, backup_stream_url FROM live_channels WHERE id = $1', [contentId]);
            if (channel.rows.length > 0) {
                streamingUrls = {
                    primary: `${config.cdn.streamingUrl}/live/${contentId}/${streamToken}.m3u8`,
                    backup: channel.rows[0].backup_stream_url ? `${config.cdn.streamingUrl}/live/${contentId}/backup/${streamToken}.m3u8` : null
                };
            }
        } else {
            // VOD content
            const contentFiles = await db.query(
                'SELECT * FROM vod_content_files WHERE content_id = $1 AND is_active = true ORDER BY quality',
                [contentId]
            );

            streamingUrls = contentFiles.rows.reduce((urls, file) => {
                urls[file.quality] = `${config.cdn.streamingUrl}/vod/${contentId}/${file.quality}/${streamToken}.m3u8`;
                return urls;
            }, {});
        }

        res.json({
            sessionId: session.rows[0].id,
            streamToken,
            streamingUrls,
            expiresAt: new Date(Date.now() + 6 * 60 * 60 * 1000) // 6 hours
        });
    } catch (error) {
        console.error('Streaming authorization error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// End streaming session
app.post('/api/streaming/end-session', authenticateToken, async (req, res) => {
    try {
        const { sessionId } = req.body;

        await db.query(
            'UPDATE streaming_sessions SET is_active = false, end_time = NOW() WHERE id = $1 AND user_id = $2',
            [sessionId, req.user.id]
        );

        res.json({ message: 'Streaming session ended' });
    } catch (error) {
        console.error('End session error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// ==========================================
// WEBSOCKET FOR REAL-TIME FEATURES
// ==========================================

wss.on('connection', (ws, req) => {
    console.log('New WebSocket connection');

    ws.on('message', async (message) => {
        try {
            const data = JSON.parse(message);
            
            if (data.type === 'authenticate') {
                const token = data.token;
                try {
                    const decoded = jwt.verify(token, config.jwtSecret);
                    ws.userId = decoded.userId;
                    ws.send(JSON.stringify({ type: 'authenticated', success: true }));
                } catch (error) {
                    ws.send(JSON.stringify({ type: 'error', message: 'Invalid token' }));
                }
            }

            if (data.type === 'heartbeat' && ws.userId) {
                // Update user's last activity
                await redis.setEx(`user_activity:${ws.userId}`, 300, Date.now()); // 5 minutes TTL
                ws.send(JSON.stringify({ type: 'heartbeat_ack' }));
            }
        } catch (error) {
            console.error('WebSocket message error:', error);
        }
    });

    ws.on('close', () => {
        console.log('WebSocket connection closed');
    });
});

// ==========================================
// HEALTH CHECK & INFO ROUTES
// ==========================================

app.get('/api/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        version: '1.0.0'
    });
});

app.get('/api/info', (req, res) => {
    res.json({
        platform: 'StreamVault Pro',
        version: '1.0.0',
        features: [
            'Live TV Streaming',
            'VOD Content',
            'Multi-device Support',
            'HD/4K Quality',
            'Secure DRM',
            'Real-time Analytics'
        ]
    });
});

// ==========================================
// ERROR HANDLING
// ==========================================

// 404 handler
app.use('*', (req, res) => {
    res.status(404).json({ error: 'Endpoint not found' });
});

// Global error handler
app.use((error, req, res, next) => {
    console.error('Unhandled error:', error);
    res.status(500).json({ error: 'Internal server error' });
});

// ==========================================
// SERVER STARTUP
// ==========================================

server.listen(config.port, () => {
    console.log(`🚀 StreamVault Pro API Server running on port ${config.port}`);
    console.log(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);
    console.log(`🔗 WebSocket server enabled`);
    console.log(`💾 Database: ${config.database.host}:${config.database.port}/${config.database.database}`);
    console.log(`📮 Redis: ${config.redis.host}:${config.redis.port}`);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
    console.log('Shutting down gracefully...');
    server.close(() => {
        db.end();
        redis.disconnect();
        process.exit(0);
    });
});

module.exports = app;