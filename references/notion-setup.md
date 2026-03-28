# Notion Database Setup

Create a Notion database to store and correlate ad analyses with performance metrics.

## Quick Setup

1. Create a new Notion database called "Ad Analyzer"
2. Copy the schema below
3. Use the database ID in the script

## Database Schema

### Properties

| Property | Type | Description |
|----------|------|---|
| **Name** | Title | Video filename or ad name |
| **Status** | Select | draft, ready, published, archived |
| **Video File** | File | Attachment: uploaded video (optional) |
| **Duration (sec)** | Number | Video length in seconds |
| **Transcript** | Text | Full transcript of audio |
| **Features** | Multi-select | Detected features (tags) |
| **Visual Features** | Text | Visual tags detected |
| **Audio Features** | Text | Audio tags detected |
| **Overall Features** | Text | Overall characteristics |
| **Date Analyzed** | Date | When analysis was done |

### Performance Metrics (Add these yourself after ad runs)

| Property | Type | Notes |
|----------|------|-------|
| **Impressions** | Number | Total views |
| **Clicks** | Number | Total clicks |
| **Conversions** | Number | Completed actions |
| **CTR** | Formula | Clicks / Impressions |
| **CPC** | Number | Cost per click |
| **ROAS** | Number | Revenue / Ad spend |
| **Cost** | Number | Total ad spend |
| **Revenue** | Number | Total revenue generated |

### Template Content (Optional)

Add columns for:
- **Ad Copy** — The text or CTA used
- **Target Audience** — Age, interest, location
- **Campaign** — Which campaign it ran in
- **Platform** — Where it ran (Facebook, Instagram, TikTok, etc.)
- **Lessons Learned** — Notes on what worked/didn't

---

## How to Correlate

Once you have 10+ analyzed ads with performance data:

### 1. Group Ads by Performance

Create filtered views:
- **High Performers:** CTR > 1.5%, ROAS > 2.0
- **Medium Performers:** CTR 0.8-1.5%, ROAS 1.0-2.0
- **Low Performers:** CTR < 0.8%, ROAS < 1.0

### 2. Compare Features

Look at what's common in high performers:
```
High Performers (ROAS > 2.0) all have:
✓ Call-to-action: 9/10 ads
✓ Human talking: 8/10 ads
✓ Product visible: 10/10 ads
✓ Emotional appeal: 6/10 ads

Low Performers (ROAS < 1.0) lack:
✗ Call-to-action: 70% missing
✗ Product visible: 40% missing
```

### 3. Update Creative Brief

**Example Brief Evolution:**

**Version 1 (Initial):**
- Show the product
- Use music
- Keep under 15 seconds

**Version 2 (After 10 analyses):**
- Show product prominently (100% in high performers)
- Include human talking (80% in high performers)
- Add explicit call-to-action (90% in high performers)
- Avoid fast pacing (20% in high performers vs 60% in low)
- Use warm colors when selling food (food ads only)

**Version 3 (After 20 analyses):**
- Product within first 3 seconds
- Real person (not animation) speaking
- Open with emotional statement or problem
- Close with specific, urgent CTA
- Fast cuts only for B2B; slower for lifestyle

---

## Database Views to Create

### 1. Performance Ranking
Sort by ROAS (descending) to see winners at a glance

### 2. Feature Analysis
Filter by single feature to see its correlation:
- View "Has Call-to-Action" sorted by ROAS
- View "Shows Food" sorted by CTR
- etc.

### 3. Timeline
Group by "Date Analyzed" to track improvements over time

### 4. By Campaign
Filter to focus on specific campaigns

---

## Automation (Optional)

Once you're comfortable, consider:
1. **Bulk upload** — Export analyses from the script as JSON, bulk-import to Notion
2. **Performance sync** — Automatically pull performance metrics from Meta Ads Manager
3. **Pattern alerts** — Get notified when high-performers share new feature combinations

For now, manual entry is fine while you're learning patterns.
