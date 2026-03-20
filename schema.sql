-- RUN THIS IN SUPABASE SQL EDITOR

-- 1. Create the predictions table
CREATE TABLE IF NOT EXISTS predictions (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id uuid REFERENCES auth.users(id),
  input_text TEXT NOT NULL,
  sentiment TEXT,
  emotion TEXT,
  confidence FLOAT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 2. Enable Row Level Security (RLS)
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- 3. Policy: Users can view their own predictions
CREATE POLICY "Users can view own predictions" 
  ON predictions FOR SELECT 
  USING (auth.uid() = user_id);

-- 4. Policy: Users can insert their own predictions
CREATE POLICY "Users can insert own predictions" 
  ON predictions FOR INSERT 
  WITH CHECK (auth.uid() = user_id);

-- 5. Policy: Public predictions are viewable by everyone (optional, for the "History" feed)
CREATE POLICY "Public predictions are viewable by everyone"
  ON predictions FOR SELECT
  USING (true);
