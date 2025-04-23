--
-- PostgreSQL database dump
--

-- Dumped from database version 14.16 (Homebrew)
-- Dumped by pg_dump version 14.16 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: pranav
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


ALTER TABLE public.alembic_version OWNER TO pranav;

--
-- Name: checkpoint_blobs; Type: TABLE; Schema: public; Owner: pranav
--

CREATE TABLE public.checkpoint_blobs (
    thread_id text NOT NULL,
    checkpoint_ns text DEFAULT ''::text NOT NULL,
    channel text NOT NULL,
    version text NOT NULL,
    type text NOT NULL,
    blob bytea
);


ALTER TABLE public.checkpoint_blobs OWNER TO pranav;

--
-- Name: checkpoint_migrations; Type: TABLE; Schema: public; Owner: pranav
--

CREATE TABLE public.checkpoint_migrations (
    v integer NOT NULL
);


ALTER TABLE public.checkpoint_migrations OWNER TO pranav;

--
-- Name: checkpoint_writes; Type: TABLE; Schema: public; Owner: pranav
--

CREATE TABLE public.checkpoint_writes (
    thread_id text NOT NULL,
    checkpoint_ns text DEFAULT ''::text NOT NULL,
    checkpoint_id text NOT NULL,
    task_id text NOT NULL,
    idx integer NOT NULL,
    channel text NOT NULL,
    type text,
    blob bytea NOT NULL,
    task_path text DEFAULT ''::text NOT NULL
);


ALTER TABLE public.checkpoint_writes OWNER TO pranav;

--
-- Name: checkpoints; Type: TABLE; Schema: public; Owner: pranav
--

CREATE TABLE public.checkpoints (
    thread_id text NOT NULL,
    checkpoint_ns text DEFAULT ''::text NOT NULL,
    checkpoint_id text NOT NULL,
    parent_checkpoint_id text,
    type text,
    checkpoint jsonb NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL
);


ALTER TABLE public.checkpoints OWNER TO pranav;

--
-- Name: exercise_templates; Type: TABLE; Schema: public; Owner: pranav
--

CREATE TABLE public.exercise_templates (
    id character varying(50) NOT NULL,
    title character varying(100) NOT NULL,
    type character varying(50) NOT NULL,
    primary_muscle_group character varying(50) NOT NULL,
    secondary_muscle_groups character varying(50)[],
    is_custom boolean,
    equipment character varying(50) NOT NULL
);


ALTER TABLE public.exercise_templates OWNER TO pranav;

--
-- Name: messages; Type: TABLE; Schema: public; Owner: pranav
--

CREATE TABLE public.messages (
    id integer NOT NULL,
    session_id character varying(36),
    role character varying(20),
    content text,
    "timestamp" timestamp without time zone,
    thumbs_up boolean
);


ALTER TABLE public.messages OWNER TO pranav;

--
-- Name: messages_id_seq; Type: SEQUENCE; Schema: public; Owner: pranav
--

CREATE SEQUENCE public.messages_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.messages_id_seq OWNER TO pranav;

--
-- Name: messages_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: pranav
--

ALTER SEQUENCE public.messages_id_seq OWNED BY public.messages.id;


--
-- Name: sessions; Type: TABLE; Schema: public; Owner: pranav
--

CREATE TABLE public.sessions (
    id character varying(36) NOT NULL,
    user_id integer,
    created_at timestamp without time zone,
    logged_out_at timestamp without time zone
);


ALTER TABLE public.sessions OWNER TO pranav;

--
-- Name: users; Type: TABLE; Schema: public; Owner: pranav
--

CREATE TABLE public.users (
    id integer NOT NULL,
    username character varying(50) NOT NULL,
    password_hash character varying(128)
);


ALTER TABLE public.users OWNER TO pranav;

--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: pranav
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.users_id_seq OWNER TO pranav;

--
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: pranav
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: messages id; Type: DEFAULT; Schema: public; Owner: pranav
--

ALTER TABLE ONLY public.messages ALTER COLUMN id SET DEFAULT nextval('public.messages_id_seq'::regclass);


--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: pranav
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: pranav
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: checkpoint_blobs checkpoint_blobs_pkey; Type: CONSTRAINT; Schema: public; Owner: pranav
--

ALTER TABLE ONLY public.checkpoint_blobs
    ADD CONSTRAINT checkpoint_blobs_pkey PRIMARY KEY (thread_id, checkpoint_ns, channel, version);


--
-- Name: checkpoint_migrations checkpoint_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: pranav
--

ALTER TABLE ONLY public.checkpoint_migrations
    ADD CONSTRAINT checkpoint_migrations_pkey PRIMARY KEY (v);


--
-- Name: checkpoint_writes checkpoint_writes_pkey; Type: CONSTRAINT; Schema: public; Owner: pranav
--

ALTER TABLE ONLY public.checkpoint_writes
    ADD CONSTRAINT checkpoint_writes_pkey PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx);


--
-- Name: checkpoints checkpoints_pkey; Type: CONSTRAINT; Schema: public; Owner: pranav
--

ALTER TABLE ONLY public.checkpoints
    ADD CONSTRAINT checkpoints_pkey PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id);


--
-- Name: exercise_templates exercise_templates_pkey; Type: CONSTRAINT; Schema: public; Owner: pranav
--

ALTER TABLE ONLY public.exercise_templates
    ADD CONSTRAINT exercise_templates_pkey PRIMARY KEY (id);


--
-- Name: messages messages_pkey; Type: CONSTRAINT; Schema: public; Owner: pranav
--

ALTER TABLE ONLY public.messages
    ADD CONSTRAINT messages_pkey PRIMARY KEY (id);


--
-- Name: sessions sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: pranav
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: pranav
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: users users_username_key; Type: CONSTRAINT; Schema: public; Owner: pranav
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);


--
-- Name: checkpoint_blobs_thread_id_idx; Type: INDEX; Schema: public; Owner: pranav
--

CREATE INDEX checkpoint_blobs_thread_id_idx ON public.checkpoint_blobs USING btree (thread_id);


--
-- Name: checkpoint_writes_thread_id_idx; Type: INDEX; Schema: public; Owner: pranav
--

CREATE INDEX checkpoint_writes_thread_id_idx ON public.checkpoint_writes USING btree (thread_id);


--
-- Name: checkpoints_thread_id_idx; Type: INDEX; Schema: public; Owner: pranav
--

CREATE INDEX checkpoints_thread_id_idx ON public.checkpoints USING btree (thread_id);


--
-- Name: messages messages_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: pranav
--

ALTER TABLE ONLY public.messages
    ADD CONSTRAINT messages_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.sessions(id);


--
-- Name: sessions sessions_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: pranav
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.users(id);


--
-- PostgreSQL database dump complete
--

