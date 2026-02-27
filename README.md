**Business Data Harbor** 

Built an AI-powered marketing assistant for Sri Lankan watch entrepreneurs.

# 🔎 1. What Problem Did this project Solve?

Identified that:

* Small entrepreneurs don’t know how to:

  * Understand competitors
  * Analyze customer reviews
  * Segment customers properly
  * Create targeted advertisements
* Large companies use data science and AI.
* Small businesses don’t have access to those tools.

So the goal was:

> Build one intelligent web system that helps business owners make data-driven marketing decisions before launching a product.

---

# 🧠 2. What Is Business Data Harbor?

It is a **web application** that combines:

* Customer segmentation
* Sentiment analysis
* Competitor analysis
* AI chatbot (LLM)
* Image generation for ads
* Vector database with RAG

All inside one system.

---

# 🏗 3. What Are the Main Components Built?

## 1️⃣ Customer Segmentation (Clustering Customers)

Used machine learning (mainly **K-Means clustering**) to:

* Group customers based on:

  * Preferences
  * Budget
  * Watch type
  * Features they like
  * Demographics

📌 Why?
So businesses can target the right group instead of marketing to everyone.

Evaluated clustering using:

* Silhouette score
* WCSS
* Elbow method

So this part is pure **unsupervised ML + evaluation**.

---

## 2️⃣ Sentiment Analysis on Reviews

Collected customer reviews and analyzed:

* Positive
* Negative
* Neutral sentiment

Why?

Because 88%+ of people read reviews before buying.

So instead of manually reading reviews, the system:

* Automatically analyzes customer emotions
* Identifies strengths and weaknesses of competitors

This helps business owners improve their product.

---

## 3️⃣ Vector Database + Embeddings

This is one of the strongest technical contributions.

Instead of using a normal database:

* Converted product data into embeddings
* Stored them inside a vector database (FAISS)
* Used similarity search to retrieve relevant data

This allows:

* Smart searching
* Semantic similarity matching
* Better context retrieval

---

## 4️⃣ RAG (Retrieval Augmented Generation)

Problem:
LLMs hallucinate and don’t know company-specific data.

Solution:
Used RAG.

Flow:

1. User asks question in chatbot
2. Question converted to embedding
3. Vector DB finds similar data
4. That data is given to LLM
5. LLM generates grounded answer

So the chatbot is **context-aware** and not generic.


## 5️⃣ AI Chatbot Interface

Built a chatbot where:

Users can ask:

* Who are my competitors?
* What do customers complain about?
* What price range works best?
* What features should I include?

Instead of dashboards, users just talk in natural language.

This makes it user-friendly for non-technical stakeholders.

---

## 6️⃣ AI Image Generation for Ads

Integrated image generation (diffusion-based models like DALL·E type systems).

Based on customer cluster, the system can:

* Generate tailored advertisement images
* Customize visuals for specific segments

This is creative AI applied to marketing.

---

# 📊 4. Data Collection

* Conducted surveys (questionnaire)
* Collected demographic data
* Collected watch preferences
* Scraped product & review data
* Combined qualitative + quantitative research

Used:

* Pragmatism research philosophy
* Mixed methods approach
* Inductive research approach
* Cross-sectional study

So academically, it is properly structured.

---

# 🏛 5. System Architecture (High-Level Flow)

Here’s what the system does step-by-step:

1. Product & review data collected
2. Data stored in:

   * SQL database (structured data)
   * Vector DB (embeddings)
3. User logs into web app
4. User selects product
5. User interacts with chatbot
6. System:

   * Retrieves relevant data
   * Runs sentiment analysis
   * Applies clustering
   * Generates AI responses
   * Optionally generates ad image

So this is a full AI pipeline system.

---

# 🧪 6. Testing & Evaluation

Tested:

* Clustering performance
* Model accuracy
* Retrieval performance
* Functional requirements
* Non-functional requirements
* Usability
* Scalability
* Security

also did:

* Self evaluation
* Stakeholder evaluation

---

# 🎯 7. Contribution

Technically:

✔ Applied vector databases in marketing
✔ Used RAG to reduce hallucination
✔ Combined ML + NLP + Generative AI in one system
✔ Built a usable business tool

Domain-wise:

✔ Helped Sri Lankan watch entrepreneurs
✔ Automated marketing research
✔ Simplified data analytics

---

# 💡 In One Powerful Summary

This project is:

> An AI-powered intelligent marketing decision support system that combines customer segmentation, sentiment analysis, vector databases, retrieval-augmented generation, and AI image generation to help Sri Lankan watch entrepreneurs make data-driven product and marketing decisions.

---

built:

🧠 A smart marketing brain
📊 That understands customers
💬 Talks like ChatGPT
📈 Analyzes competitors
🎨 Creates advertisements
📂 Searches knowledge intelligently

All inside one web system.
