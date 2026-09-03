-- Run this AFTER deploying the updated backend/frontend.
-- Drops the tables that belonged to the removed "Materials" system
-- (app/materials.py) and the removed "Study Deck" (app/study.py).
-- Personas, conversations, messages, and persona_chunks are untouched —
-- those power the new student-facing persona creation/chat flow.

drop table if exists material_quiz_attempts;
drop table if exists material_quizzes;
drop table if exists material_chunks;
drop table if exists material_messages;
drop table if exists materials;

drop table if exists study_messages;
drop table if exists study_conversations;
