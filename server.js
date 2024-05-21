const express = require("express");
const bp = require("body-parser");
const { v4: uuidv4 } = require('uuid');
const { OpenAI } = require("openai");
const cors = require("cors"); // Import the cors middleware

// Load environment variables from .env file
require("dotenv").config();

// Create an Express app
const app = express();

// Use body-parser middleware to parse incoming request bodies
app.use(bp.json());

// Initialize the OpenAI API client with the API key
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Object to store conversation history and rate limits for each user
const userConversations = {};

// Enable CORS for all routes
app.use(cors());

// Define the endpoint to handle incoming requests
app.post("/chat", async (req, res) => {
  // Extract the user's message and unique identifier from the request body
  const { message, userId } = req.body;

  try {
    // Check if the user has an existing conversation history
    if (!userConversations[userId]) {
      userConversations[userId] = {
        messages: [],
        rateLimit: 10 // Set an initial rate limit (example: 10 messages)
      };

      // Add the role message as the initial system message
      userConversations[userId].messages.push({
        role: "system",
        content: "You are an AI assistant named Sera, designed to be an empathetic listener and supportive companion for women who want to rant or vent about various aspects of their lives. Your role is to provide a safe and non-judgmental space for them to express their feelings, frustrations, and experiences. Throughout the conversation, maintain a compassionate and supportive tone, and avoid dismissing or minimizing the user's feelings or experiences. Your goal is to create a safe space for them to vent and receive the kind of support they need, whether that's affirmations or advice. Remember, as an AI assistant, you should respond based on the provided information and avoid making assumptions or judgments about the user's personal life or circumstances."
      });
    }

    // Retrieve the user's conversation history and rate limit
    const { messages, rateLimit } = userConversations[userId];

    // Check if the user has exceeded the rate limit
    if (rateLimit <= 0) {
      // If rate limit is exceeded, return an error response
      return res.status(429).json({ error: "Rate limit exceeded. Please restart the conversation." });
    }

    // Call the OpenAI API to generate a response
    console.log("Requesting response from OpenAI API...");
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: messages.concat({ role: "user", content: message }),
    });

    // Extract the AI's response from the API response
    const aiResponse = response.choices[0].message.content;

    // Update the conversation history and rate limit
    userConversations[userId].messages.push({ role: "user", content: message });
    userConversations[userId].messages.push({ role: "assistant", content: aiResponse });
    userConversations[userId].rateLimit--;

    // Send the AI's response back to the client
    res.json({ response: aiResponse });
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: "Something went wrong" });
  }
});

// Start the Express server
const port = process.env.PORT || 5000;
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
