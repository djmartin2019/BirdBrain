CREATE TABLE predictions {
    id SERIAL PRIMARY KEY,
    filename TEXT,
    predicted_species TEXT NOT NULL,
    confidence NUMERIC NOT NULL
    top_5 JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
};
