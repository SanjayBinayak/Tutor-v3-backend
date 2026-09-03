-- Additive migration — run this once against your Supabase project, after
-- schema_v1 through schema_v4_quiz. Nothing here alters or drops any
-- existing table.
--
-- Adds:
--   1. persona_chunks + match_persona_chunks() — lets /study/ask and
--      /personas/{id}/ask retrieve only the solved-question records
--      relevant to a student's question instead of resending a teacher's
--      entire solved_questions blob on every call (see app/persona_rag.py).
--      Mirrors the material_chunks / match_material_chunks pattern already
--      used for Document Chat in app/materials.py.
--   2. study_conversations + study_messages — Study Deck chat history
--      (ChatGPT-style: list past chats, reopen one, keep asking follow-ups
--      inside it), mirroring the material_messages pattern.

create extension if not exists vector;

-- ---------------------------------------------------------------------
-- Persona reference-material RAG
-- ---------------------------------------------------------------------
create table if not exists persona_chunks (
    id uuid primary key default gen_random_uuid(),
    persona_id uuid not null references personas(id) on delete cascade,
    chunk_index integer not null,
    section text not null default 'solved_questions',
    content text not null,
    -- Must match app/config.py's MATERIAL_EMBEDDING_DIM (also used here —
    -- embeddings are cheap/free-tier-generous, no need for a second model).
    embedding vector(768),
    created_at timestamptz not null default now()
);

create index if not exists idx_persona_chunks_persona on persona_chunks(persona_id);
create index if not exists persona_chunks_embedding_idx on persona_chunks
    using ivfflat (embedding vector_cosine_ops) with (lists = 100);

alter table persona_chunks enable row level security;
-- No public policies — same reasoning as material_chunks: the backend
-- talks to Supabase with the service-role key (bypasses RLS), so leaving
-- this table policy-less just means no anon/direct client can read it.

create or replace function match_persona_chunks(
    p_persona_id uuid,
    p_query_embedding vector(768),
    p_match_count int default 6
)
returns table (
    id uuid,
    content text,
    section text,
    chunk_index integer,
    similarity float
)
language sql stable
as $$
    select
        pc.id,
        pc.content,
        pc.section,
        pc.chunk_index,
        1 - (pc.embedding <=> p_query_embedding) as similarity
    from persona_chunks pc
    where pc.persona_id = p_persona_id
    order by pc.embedding <=> p_query_embedding
    limit p_match_count;
$$;


-- ---------------------------------------------------------------------
-- Study Deck chat history
-- ---------------------------------------------------------------------
create table if not exists study_conversations (
    id uuid primary key default gen_random_uuid(),
    student_id uuid not null references auth.users(id) on delete cascade,
    persona_id uuid references personas(id) on delete set null,
    title text not null default 'New chat',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);
create index if not exists idx_study_conversations_student
    on study_conversations(student_id, updated_at desc);
alter table study_conversations enable row level security;
create policy "Students see their own study conversations"
    on study_conversations for select
    using (auth.uid() = student_id);

create table if not exists study_messages (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid not null references study_conversations(id) on delete cascade,
    role text not null check (role in ('student', 'tutor')),
    content text not null default '',
    -- Only populated for 'tutor' rows: the full {section_key: content}
    -- dict from that /study/ask call, so reopening a past chat can
    -- re-render every tab, not just the first section.
    sections jsonb,
    created_at timestamptz not null default now()
);
create index if not exists idx_study_messages_conversation
    on study_messages(conversation_id, created_at);
alter table study_messages enable row level security;
create policy "Students see messages in their own study conversations"
    on study_messages for select
    using (
        conversation_id in (select id from study_conversations where student_id = auth.uid())
    );
