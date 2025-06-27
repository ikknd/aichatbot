import 'dotenv/config';
import * as cheerio from 'cheerio';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { createClient } from '@supabase/supabase-js';
import OpenAI from 'openai';

const BASE_URL = process.env.DOCS_URL;
// const LANGUAGES = ['en', 'es', 'nl', 'fr', 'it', 'de'];
const LANGUAGES = ['en'];

// Initialize Supabase client
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

// Embedding model
async function getEmbedding(text) {
    try {
        const openai = new OpenAI({
            baseURL: process.env.LLM_EMBEDDING_URL,
            apiKey: process.env.LLM_KEY,
        });

      const response = await openai.embeddings.create({
        model: process.env.LLM_EMBEDDING_MODEL,
        input: text,
        dimensions: 768,
      });
      
      return response.data[0].embedding;
    } catch (error) {
      console.error("Error getting embedding:", error.response ? error.response.data : error.message);
      throw error;
    }
}

async function storeChunkInSupabase(chunk) {
    try {
        const { data, error } = await supabase
            .from('article_chunks')
            .insert({
                content: chunk.content,
                source_url: chunk.metadata.source_url,
                title: chunk.metadata.title,
                chunk_index: chunk.metadata.chunk_index,
                total_chunks: chunk.metadata.total_chunks,
                embedding: chunk.embedding
            })
            .select();

        if (error) throw error;
        return data;
    } catch (error) {
        console.error('Error storing chunk in Supabase:', error);
        return null;
    }
}

async function fetchPage(url) {
    try {
        const response = await fetch(url, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
            }
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const html = await response.text();
        console.log(`Successfully fetched ${url}`);
        return html;
    } catch (error) {
        console.error(`Error fetching ${url}:`, error);
        return null;
    }
}

async function getSectionLinks(language) {
    const html = await fetchPage(`${BASE_URL}/${language}`);
    if (!html) return [];

    const $ = cheerio.load(html);
    const sectionLinks = [];
    
    $('.kb-index a').each((_, element) => {
        const href = $(element).attr('href');
        if (href) {
            sectionLinks.push(`${BASE_URL}${href}`);
        }
    });

    return sectionLinks;
}

async function getArticleLinks(sectionUrl) {
    const html = await fetchPage(sectionUrl);
    if (!html) return [];

    const $ = cheerio.load(html);
    const articleLinks = $('.kb-categories__item ul li a').map((_, el) => $(el).attr('href')).get();
    
    return articleLinks;
}

async function getArticleContent(articleUrl) {
    const html = await fetchPage(articleUrl);
    if (!html) return null;

    const $ = cheerio.load(html);
    const articleDiv = $('.kb-article.tinymce-content');
    
    if (articleDiv.length === 0) {
        console.log('Article content div not found');
        return null;
    }

    // Get article title
    const title = $('h1 span').text().trim();
    articleDiv.find('h1').remove();

    // Remove all images and videos
    articleDiv.find('img').remove();
    articleDiv.find('video').remove();
    
    // Get text content and clean it
    let content = articleDiv.text()
        .replace(/\s+/g, ' ')  // Replace multiple spaces with single space
        .trim();               // Remove leading/trailing whitespace

    return { content, title };
}

async function splitTextIntoChunks(text) {
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 100,
        separators: ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    });

    const chunks = await textSplitter.splitText(text);
    return chunks;
}

async function processArticle(articleUrl, articleIndex, totalArticles) {
    console.log(`\nProcessing article ${articleIndex + 1}/${totalArticles}`);
    console.log(`URL: ${articleUrl}`);
    
    const { content, title } = await getArticleContent(articleUrl);
    if (!content) {
        console.log('Failed to get article content');
        return 0;
    }

    const chunks = await splitTextIntoChunks(content);
    console.log(`Split article into ${chunks.length} chunks`);
    
    let storedCount = 0;
    for (let i = 0; i < chunks.length; i++) {
        const chunk = title + " " + chunks[i];
        console.log(`Processing chunk ${i + 1}/${chunks.length}`);
        
        // Get embedding for the chunk
        const embedding = await getEmbedding(chunk);
        
        const chunkData = {
            content: chunk,
            metadata: {
                source_url: articleUrl,
                title: title,
                chunk_index: i,
                total_chunks: chunks.length
            },
            embedding: embedding
        };

        // Store in Supabase
        const result = await storeChunkInSupabase(chunkData);
        if (result) storedCount++;
    }
    
    console.log(`Successfully stored ${storedCount} chunks for this article`);
    return storedCount;
}

async function main() {
    console.log('Starting to scrape Documentation site...');
    
    for (const language of LANGUAGES) {
        console.log(`\n=== Processing language: ${language} ===`);
        
        // Get all section links for this language
        const sectionLinks = await getSectionLinks(language);
        console.log(`Found ${sectionLinks.length} sections for ${language}`);
        
        // Get all article links from each section
        const allArticles = [];
        for (const sectionUrl of sectionLinks) {
            console.log(`\nProcessing section: ${sectionUrl}`);
            const articles = await getArticleLinks(sectionUrl);
            console.log(`Found ${articles.length} articles in this section`);
            allArticles.push(...articles);
        }
        
        console.log(`\nTotal articles found: ${allArticles.length}`);
        
        // Process all articles
        let totalStoredChunks = 0;
        for (let i = 0; i < allArticles.length; i++) {
            const storedChunks = await processArticle(allArticles[i], i, allArticles.length);
            totalStoredChunks += storedChunks;
        }
        
        console.log(`\n=== Summary for ${language} ===`);
        console.log(`Total articles processed: ${allArticles.length}`);
        console.log(`Total chunks stored: ${totalStoredChunks}`);
    }
}

main().catch(console.error); 