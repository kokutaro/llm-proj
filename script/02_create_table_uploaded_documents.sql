-- Drop table
DROP TABLE IF EXISTS public.uploaded_documents;

-- Create table
CREATE TABLE public.uploaded_documents (
	uuid uuid NOT NULL,
	file_name varchar NOT NULL,
	sha512 varchar NOT NULL,
	content_type varchar NOT NULL,
	CONSTRAINT uploaded_documents_pkey PRIMARY KEY (uuid)
);
CREATE UNIQUE INDEX uploaded_documents_sha512_idx ON public.uploaded_documents USING btree (sha512);