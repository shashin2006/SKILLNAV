const express = require('express');
const multer = require('multer');
const pdf = require('pdf-parse');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
const PORT = process.env.PORT || 3000;

// Enhanced CORS configuration
app.use(cors({
    origin: ['http://localhost:3000', 'http://localhost:5500', 'http://127.0.0.1:5500'],
    credentials: true
}));

app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ extended: true, limit: '100mb' }));

// Serve static files from multiple directories
app.use(express.static(path.join(__dirname, '..'))); // Serve from root
app.use(express.static(__dirname)); // Serve from project folder

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = path.join(__dirname, 'uploads');
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir, { recursive: true });
        }
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + '-' + file.originalname);
    }
});

const upload = multer({ 
    storage: storage,
    fileFilter: (req, file, cb) => {
        if (file.mimetype === 'application/pdf') {
            cb(null, true);
        } else {
            cb(new Error('Only PDF files are allowed'), false);
        }
    },
    limits: {
        fileSize: 25 * 1024 * 1024 // Increased to 25MB
    }
});

// Initialize Google Gemini AI
let genAI;
try {
    if (!process.env.GOOGLE_API_KEY) {
        console.warn('⚠️  GOOGLE_API_KEY is not set in environment variables');
    } else {
        genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
        console.log('✅ Gemini AI initialized successfully');
    }
} catch (error) {
    console.error('❌ Error initializing Gemini AI:', error.message);
}

async function getGeminiResponse(input, pdfContent, prompt) {
    try {
        if (!genAI) {
            throw new Error('Gemini AI not initialized. Please check your GOOGLE_API_KEY in .env file');
        }

        const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
        
        // Remove content limits - let Gemini handle the full content
        const fullPrompt = `
        ${prompt}
        
        ${input ? `Job Description: ${input}` : ''}
        
        Resume Content: ${pdfContent}
        `;

        console.log('📤 Sending request to Gemini AI...');
        const result = await model.generateContent(fullPrompt);
        const response = await result.response;
        console.log('✅ Received response from Gemini AI');
        return response.text();
    } catch (error) {
        console.error('❌ Error calling Gemini API:', error);
        if (error.message.includes('API_KEY_INVALID')) {
            throw new Error('Invalid Google API Key. Please check your GOOGLE_API_KEY in .env file');
        } else if (error.message.includes('Quota exceeded')) {
            throw new Error('API quota exceeded. Please check your Google AI Studio quota.');
        } else if (error.message.includes('SAFETY')) {
            throw new Error('Content was blocked for safety reasons. Please try with different content.');
        } else if (error.message.includes('content too long')) {
            // If content is too long, try with a smaller chunk
            console.log('Content too long, trying with first 40000 characters...');
            const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
            const fullPrompt = `
            ${prompt}
            
            ${input ? `Job Description: ${input}` : ''}
            
            Resume Content: ${pdfContent.substring(0, 40000)}
            `;
            const result = await model.generateContent(fullPrompt);
            const response = await result.response;
            return response.text();
        } else {
            throw new Error(`AI Service Error: ${error.message}`);
        }
    }
}

async function inputPdfText(filePath) {
    try {
        console.log('📄 Reading PDF file:', filePath);
        
        if (!fs.existsSync(filePath)) {
            throw new Error('PDF file not found');
        }

        const dataBuffer = fs.readFileSync(filePath);
        const data = await pdf(dataBuffer);
        
        if (!data.text || data.text.trim().length === 0) {
            throw new Error('No text could be extracted from the PDF. The PDF might be scanned or image-based.');
        }
        
        console.log('✅ PDF text extracted successfully, length:', data.text.length);
        return data.text;
    } catch (error) {
        console.error('❌ Error reading PDF:', error);
        throw new Error(`PDF Processing Error: ${error.message}`);
    }
}

// Prompts
const input_prompt1 = `
You are an experienced Technical Human Resource Manager,your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
`;

const input_prompt2 = `
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to evaluate the resume against the provided job description. As a Human Resource manager,
assess the compatibility of the resume with the role. Give me what are the keywords that are missing
Also, provide recommendations for enhancing the candidate's skills and identify which areas require further development.
`;

const input_prompt3 = `
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
the job description. First the output should come as percentage and then keywords missing and last final thoughts.
`;

const input_prompt4 = `
You are a seasoned Technical Human Resource Manager and an expert in Applicant Tracking Systems (ATS), 
tasked with performing a comprehensive evaluation of the provided resume against the job description.

Please analyze and deliver:
1. **Resume Overview**: Assess the candidate's profile strengths and weaknesses in relation to the role.
2. **Keyword Matching**: Identify any missing keywords that may enhance the candidate's alignment with the job.
3. **Match Percentage**: Provide a calculated percentage indicating how well the resume matches the job description.
4. **Final Thoughts**: Offer professional recommendations for resume improvement, detailing areas for skill enhancement.

Ensure a structured and thorough response to guide the candidate effectively.
`;

// Store analysis sessions in memory (in production, use Redis or database)
const analysisSessions = new Map();

// API Routes
// Add this to your existing server.js - Roadmap Generation Endpoint
app.post('/api/generate-roadmap', async (req, res) => {
    try {
        const { domain } = req.body;
        
        if (!domain) {
            return res.status(400).json({ 
                success: false, 
                error: 'Domain is required' 
            });
        }

        console.log('🛣️  Generating roadmap for domain:', domain);

        const prompt = `
        Create a comprehensive 4-level career roadmap for the domain: ${domain}

        Return ONLY valid JSON in this exact format:
        {
            "roadmap": [
                {
                    "level": 0,
                    "title": "Foundation Level (Beginner)",
                    "nodes": [
                        {
                            "title": "Specific Skill Name",
                            "content": "Clear description of what to learn and practice",
                            "duration": "Realistic timeframe like '1-2 months'"
                        }
                    ]
                },
                {
                    "level": 1, 
                    "title": "Intermediate Level",
                    "nodes": [
                        {
                            "title": "Specific Skill Name",
                            "content": "Clear description building on foundation skills",
                            "duration": "Realistic timeframe"
                        }
                    ]
                },
                {
                    "level": 2,
                    "title": "Advanced Level", 
                    "nodes": [
                        {
                            "title": "Specific Skill Name",
                            "content": "Advanced topics and specialized knowledge",
                            "duration": "Realistic timeframe"
                        }
                    ]
                },
                {
                    "level": 3,
                    "title": "Specialization Level",
                    "nodes": [
                        {
                            "title": "Career Specialization",
                            "content": "Specific career path or specialization area",
                            "duration": "Long-term development"
                        }
                    ]
                }
            ]
        }

        Requirements:
        - Create 3-4 nodes for each level
        - Focus on practical, industry-relevant skills for ${domain}
        - Include both technical and essential soft skills
        - Make progression logical from beginner to expert
        - Consider current industry trends for ${domain}
        - Ensure durations are realistic for effective learning
        - Include modern tools and technologies specific to ${domain}
        `;

        const response = await getGeminiResponse(domain, prompt);
        
        // Parse the JSON response from Gemini
        let roadmap;
        try {
            // Extract JSON from the response
            const jsonMatch = response.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                roadmap = JSON.parse(jsonMatch[0]);
            } else {
                roadmap = JSON.parse(response);
            }
            
            // Validate the roadmap structure
            if (!roadmap.roadmap || !Array.isArray(roadmap.roadmap)) {
                throw new Error('Invalid roadmap structure received from AI');
            }
            
        } catch (parseError) {
            console.error('Error parsing AI response:', parseError);
            console.log('Raw AI response:', response);
            
            // Fallback roadmap if parsing fails
            roadmap = createFallbackRoadmap(domain);
        }

        res.json({ 
            success: true, 
            roadmap: roadmap.roadmap
        });

    } catch (error) {
        console.error('❌ Error generating roadmap:', error);
        res.status(500).json({ 
            success: false, 
            error: error.message 
        });
    }
});

// Fallback roadmap generator (add this function too)
function createFallbackRoadmap(domain) {
    return {
        roadmap: [
            {
                level: 0,
                title: "Foundation Level",
                nodes: [
                    {
                        title: "Basic Concepts",
                        content: `Learn fundamental concepts and principles of ${domain}`,
                        duration: "1-2 months"
                    },
                    {
                        title: "Core Tools",
                        content: "Get familiar with essential tools and technologies",
                        duration: "1-2 months"
                    },
                    {
                        title: "Basic Projects",
                        content: "Build simple projects to apply foundational knowledge",
                        duration: "1 month"
                    }
                ]
            },
            {
                level: 1,
                title: "Intermediate Level",
                nodes: [
                    {
                        title: "Advanced Concepts",
                        content: "Dive deeper into core concepts and methodologies",
                        duration: "2-3 months"
                    },
                    {
                        title: "Real-world Applications",
                        content: "Work on more complex, practical applications",
                        duration: "2 months"
                    },
                    {
                        title: "Industry Tools",
                        content: "Master industry-standard tools and frameworks",
                        duration: "2 months"
                    }
                ]
            },
            {
                level: 2,
                title: "Advanced Level",
                nodes: [
                    {
                        title: "Specialized Topics",
                        content: "Focus on specialized areas within the domain",
                        duration: "3-4 months"
                    },
                    {
                        title: "Complex Projects",
                        content: "Build comprehensive, real-world projects",
                        duration: "3 months"
                    },
                    {
                        title: "Best Practices",
                        content: "Learn industry best practices and standards",
                        duration: "2 months"
                    }
                ]
            },
            {
                level: 3,
                title: "Specialization Level",
                nodes: [
                    {
                        title: "Career Path",
                        content: "Choose and deep dive into specific career specialization",
                        duration: "6+ months"
                    },
                    {
                        title: "Industry Trends",
                        content: "Stay updated with latest trends and technologies",
                        duration: "Ongoing"
                    },
                    {
                        title: "Professional Development",
                        content: "Focus on soft skills and professional growth",
                        duration: "Continuous"
                    }
                ]
            }
        ]
    };
}
app.post('/api/analyze-resume', upload.single('resume'), async (req, res) => {
    let filePath = null;
    
    try {
        console.log('📨 Received resume analysis request');
        console.log('Analysis type:', req.body.analysisType);
        console.log('Job description present:', !!req.body.jobDescription);
        console.log('Custom query present:', !!req.body.customQuery);
        console.log('File received:', req.file ? req.file.originalname : 'No file');

        if (!req.file) {
            console.log('❌ No file uploaded');
            return res.status(400).json({ 
                success: false, 
                error: 'No PDF file uploaded. Please select a PDF file.' 
            });
        }

        filePath = req.file.path;
        console.log('File saved at:', filePath);

        // Extract text from PDF
        const pdfContent = await inputPdfText(filePath);
        console.log('PDF content extracted, length:', pdfContent.length);
        
        const { jobDescription, analysisType, customQuery } = req.body;
        console.log('Job description length:', jobDescription ? jobDescription.length : 0);
        console.log('Analysis type requested:', analysisType);

        let prompt;
        switch (analysisType) {
            case 'overview':
                prompt = input_prompt1;
                break;
            case 'keywords':
                prompt = input_prompt2;
                break;
            case 'match':
                prompt = input_prompt3;
                break;
            case 'all':
                prompt = input_prompt4;
                break;
            case 'query':
                prompt = customQuery || 'Please analyze this resume';
                break;
            default:
                prompt = input_prompt1;
        }

        console.log('Using prompt for:', analysisType);

        // Get AI response
        const input = analysisType === 'query' ? customQuery : jobDescription;
        console.log('🔍 Calling Gemini AI...');
        console.log('Input length:', input ? input.length : 0);
        
        const response = await getGeminiResponse(input, pdfContent, prompt);
        console.log('✅ Analysis completed successfully');
        console.log('Response length:', response.length);

        // Create session for results page
        const sessionId = Date.now().toString() + Math.random().toString(36).substr(2, 9);
        analysisSessions.set(sessionId, {
            analysis: response,
            type: analysisType,
            timestamp: Date.now(),
            fileName: req.file.originalname
        });

        // Clean up old sessions (older than 1 hour)
        const oneHourAgo = Date.now() - 60 * 60 * 1000;
        for (let [id, session] of analysisSessions.entries()) {
            if (session.timestamp < oneHourAgo) {
                analysisSessions.delete(id);
            }
        }

        res.json({ 
            success: true, 
            sessionId: sessionId,
            type: analysisType,
            analysis: response 
        });

    } catch (error) {
        console.error('❌ Error in resume analysis:', error);
        console.error('Error stack:', error.stack);
        res.status(500).json({ 
            success: false, 
            error: error.message 
        });
    } finally {
        // Clean up uploaded file
        if (filePath && fs.existsSync(filePath)) {
            try {
                fs.unlinkSync(filePath);
                console.log('🧹 Cleaned up file:', filePath);
            } catch (cleanupError) {
                console.error('Error cleaning up file:', cleanupError);
            }
        }
    }
});

// Get analysis results by session ID
app.get('/api/analysis-results/:sessionId', (req, res) => {
    const { sessionId } = req.params;
    const session = analysisSessions.get(sessionId);
    
    if (!session) {
        return res.status(404).json({
            success: false,
            error: 'Analysis results not found or expired'
        });
    }

    res.json({
        success: true,
        analysis: session.analysis,
        type: session.type,
        fileName: session.fileName,
        timestamp: session.timestamp
    });
});

// Health check endpoint with detailed info
app.get('/api/health', (req, res) => {
    const healthInfo = {
        status: 'OK',
        message: 'Server is running',
        timestamp: new Date().toISOString(),
        gemini_configured: !!process.env.GOOGLE_API_KEY,
        port: PORT,
        environment: process.env.NODE_ENV || 'development'
    };

    if (!process.env.GOOGLE_API_KEY) {
        healthInfo.warning = 'GOOGLE_API_KEY is not set in .env file';
    }

    res.json(healthInfo);
});

app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}/login.html`);
    console.log(`❤️  Health Check: http://localhost:${PORT}/api/health`);
    
    if (!process.env.GOOGLE_API_KEY) {
        console.warn('⚠️  WARNING: GOOGLE_API_KEY is not set in .env file!');
        console.warn('   Create a .env file in the project folder with:');
        console.warn('   GOOGLE_API_KEY=your_actual_api_key_here');
    } else {
        console.log('✅ Google API Key is configured');
    }
});