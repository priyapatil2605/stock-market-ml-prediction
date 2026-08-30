# Vibe Closet

A personal styling app that recommends outfits from your own wardrobe based on aesthetic/vibe (Old Money, Y2K, Streetwear, Clean Girl, Italian Coastal, Minimalist), with basic fit and sustainability guidance.

> Status: early build. Frontend mockup complete with static data. Backend, database, and ML matching are in progress.

---

## Why this project

Most fashion apps solve one problem in isolation — a closet organizer, a virtual try-on tool, or a size guide — but rarely combine them. Vibe Closet's goal is to bring wardrobe management, aesthetic-based outfit recommendation, and basic fit/sustainability info into a single simple flow, without the complexity (or cost) of a full virtual try-on system.

---

## Features

### 1. Wardrobe
Upload and organize your clothing items with tags for category, color, occasion, and season. This is the core data every other feature builds on.

### 2. Vibe / Aesthetic Matching
Pick a style identity — Old Money, Y2K, Streetwear, Clean Girl, Italian Coastal, Minimalist — and get outfit suggestions styled to that aesthetic using items you actually own.

### 3. Outfit Generator
Combines wardrobe items by category (top / bottom / shoes / accessories) filtered by selected vibe and occasion into a ready-to-wear outfit suggestion.

### 4. Size / Fit Checker
A simple brand-based lookup that tells you whether a brand tends to run small, true-to-size, or large for a given category, to reduce fit guesswork before buying.

### 5. Sustainability Tags
Each wardrobe item gets a rough sustainability rating (high / medium / low) based on its listed material, so users can see the environmental footprint of their closet at a glance.

---

## Tech Stack

| Layer | Tech |
|---|---|
| Frontend (web) | React |
| Frontend (mobile) | React Native *(planned)* |
| Backend | Node.js + Express |
| Database | MongoDB |
| Image storage | Cloudinary |
| Vibe matching (ML) | CLIP embeddings + cosine similarity *(planned)* |
| Hosting | Vercel (frontend), Render (backend) *(planned)* |

---

## Project Status

| Module | Status |
|---|---|
| Static UI mockup (all 5 features, fake data) | ✅ Done |
| Backend server + routes | 🔲 Not started |
| Database schema + connection | 🔲 Not started |
| Wardrobe CRUD (real data) | 🔲 Not started |
| Auth (signup/login) | 🔲 Not started |
| Vibe matching via CLIP | 🔲 Not started |
| Outfit generator (real logic) | 🔲 Not started |
| Size checker (static lookup) | 🔲 Not started |
| Sustainability tags (static lookup) | 🔲 Not started |
| Mobile app | 🔲 Not started |
| Deployment | 🔲 Not started |

---

## Planned Roadmap (v2 and beyond)

These are intentionally out of scope for v1 due to complexity, but are natural next steps:

- **Virtual try-on** — overlay outfit combinations on a user photo using pose estimation / diffusion-based try-on models
- **Trend prediction** — surface trending styles by analyzing social platforms and re-ranking outfit suggestions accordingly
- **ML-based size prediction** — replace the static brand lookup with a model trained on user measurements and return data

---

## Getting Started

*(To be filled in once the backend is set up — will include install steps, environment variables, and how to run locally.)*

---

## Author

Built by [Your Name], 3rd year Computer Science, as a full-stack + ML learning project.
