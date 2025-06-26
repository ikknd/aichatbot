import 'dotenv/config';
import { pipeline } from '@xenova/transformers';
import { createClient } from '@supabase/supabase-js';
import OpenAI from 'openai';

// Initialize Supabase client
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

// Embedding model
let embedder = null;
async function getEmbedder() {
    if (!embedder) {
        console.log('Initializing embedding model...');
        embedder = await pipeline('feature-extraction', 'Xenova/bge-small-en');
        console.log('Embedding model initialized');
    }
    return embedder;
}

async function getEmbedding(text) {
    const extractor = await getEmbedder();
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
}

async function querySupabase(queryEmbedding, topK = 5) {
    const { data, error } = await supabase.rpc('match_article_chunks', {
        query_embedding: queryEmbedding,
        match_count: topK
    });
    if (error) {
        console.error('Error querying Supabase:', error);
        return [];
    }
    return data;
}

async function main() {
    const userQuery = process.argv.slice(2).join(' ');
    if (!userQuery) {
        console.error('Please provide a query as a command line argument.');
        process.exit(1);
    }
    console.log('User query:', userQuery);

    // 1. Generate embedding for the query
    const queryEmbedding = await getEmbedding(userQuery);

    // 2. Query Supabase for similar chunks
    const topK = 5;
    const results = await querySupabase(queryEmbedding, topK);
    console.log('Results: ', results.length);
    
    // 3. Prepare context string from top chunks
    const context = results.map((row, i) => `Context ${i + 1} (from: ${row.title}):\n${row.content}`).join('\n\n');

    // 4. Prepare messages for LLM
    const systemPrompt = `You are a helpful customer support chatbot. 
                            Use the provided context to answer the user's question as 
                            accurately as possible. If the answer is not in the context, 
                            say you don't know. Prefer structured answers. 
                            Do not make up information. Do not use technical jargon, 
                            or variable names passed to you.
                            `;
    const messages = [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: `Context:
${context}

User question: ${userQuery}` }
    ];

    // 5. Call local LLM using OpenAI SDK
    const openai = new OpenAI({
        baseURL: process.env.LLM_URL,
        apiKey: process.env.LLM_KEY,
    });

    const response = await openai.chat.completions.create({
        model: process.env.LLM_MODEL,
        messages,
        max_tokens: 512,
        temperature: 0.2
    });

    // 6. Print LLM response
    console.log('\n--- AI Response ---');
    console.log(response.choices[0].message.content);
}

main().catch(console.error);
