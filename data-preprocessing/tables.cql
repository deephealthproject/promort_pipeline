//////////////////////////////////////////////////////////////////////
// Oskar's new slides

// DROP TABLE promort.ids_cosk_0; DROP TABLE promort.data_cosk_0; DROP TABLE promort.metadata_cosk_0;
// TRUNCATE TABLE promort.ids_cosk_0; TRUNCATE TABLE promort.data_cosk_0; TRUNCATE TABLE promort.metadata_cosk_0;

// Mapping patch metadata to data via uuid
CREATE TABLE promort.ids_cosk_0(
  sample_name text, // AN1234
  sample_rep int, // 5
  x int, 
  y int,
  label int,
  patch_id uuid,
  tissue float,
  PRIMARY KEY ((sample_name, sample_rep, label), x, y)
);

// Mapping patch uuid to data and label
CREATE TABLE promort.data_cosk_0(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE promort.metadata_cosk_0(
  sample_name text, // AN1234
  sample_rep int, // 5
  x int, 
  y int,
  label int,
  patch_id uuid,
  tissue float,
  PRIMARY KEY ((patch_id))
);

///// level 1

// DROP TABLE promort.ids_cosk_1; DROP TABLE promort.data_cosk_1; DROP TABLE promort.metadata_cosk_1;
// TRUNCATE TABLE promort.ids_cosk_1; TRUNCATE TABLE promort.data_cosk_1; TRUNCATE TABLE promort.metadata_cosk_1;


// Mapping patch metadata to data via uuid
CREATE TABLE promort.ids_cosk_1(
  sample_name text, // AN1234
  sample_rep int, // 5
  x int, 
  y int,
  label int,
  patch_id uuid,
  tissue float,
  PRIMARY KEY ((sample_name, sample_rep, label), x, y)
);

// Mapping patch uuid to data and label
CREATE TABLE promort.data_cosk_1(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE promort.metadata_cosk_1(
  sample_name text, // AN1234
  sample_rep int, // 5
  x int, 
  y int,
  label int,
  patch_id uuid,
  tissue float,
  PRIMARY KEY ((patch_id))
);

///// level 2

// DROP TABLE promort.ids_cosk_2; DROP TABLE promort.data_cosk_2; DROP TABLE promort.metadata_cosk_2;
// TRUNCATE TABLE promort.ids_cosk_2; TRUNCATE TABLE promort.data_cosk_2; TRUNCATE TABLE promort.metadata_cosk_2;


// Mapping patch metadata to data via uuid
CREATE TABLE promort.ids_cosk_2(
  sample_name text, // AN1234
  sample_rep int, // 5
  x int, 
  y int,
  label int,
  patch_id uuid,
  tissue float,
  PRIMARY KEY ((sample_name, sample_rep, label), x, y)
);

// Mapping patch uuid to data and label
CREATE TABLE promort.data_cosk_2(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE promort.metadata_cosk_2(
  sample_name text, // AN1234
  sample_rep int, // 5
  x int, 
  y int,
  label int,
  patch_id uuid,
  tissue float,
  PRIMARY KEY ((patch_id))
);


/// Promort slides with metadata

// Mapping patch metadata to data via uuid
CREATE TABLE promort.ids_0(
  sample_name text, // AN1234
  sample_rep int, // 5
  x int, 
  y int,
  label int,
  patch_id uuid,
  PRIMARY KEY ((sample_name, sample_rep, label), x, y)
);

// Mapping patch uuid to data and label
CREATE TABLE promort.data_0(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE promort.metadata_0(
  sample_name text, // AN1234
  sample_rep int, // 5
  x int, 
  y int,
  label int,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);
