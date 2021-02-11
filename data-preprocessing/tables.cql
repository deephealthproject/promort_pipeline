//////////////////////////////////////////////////////////////////////
// Model 1

CREATE TABLE promort.patches(
  sample_name text, // AN1234
  sample_rep int, // 5
  x int, 
  y int,
  label int,
  data blob,
  varia int,
  PRIMARY KEY ((sample_name, label), sample_rep, x, y)
);

//////////////////////////////////////////////////////////////////////
// Model 2

// Mapping patch metadata to data via uuid
CREATE TABLE promort.ids_by_metadata(
  sample_name text, // AN1234
  sample_rep int, // 5
  x int, 
  y int,
  label int,
  patch_id uuid,
  PRIMARY KEY ((sample_name, label), sample_rep, x, y)
);

// Mapping patch uuid to data and label
CREATE TABLE promort.data_by_ids(
  patch_id uuid,
  label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

//////////////////////////////////////////////////////////////////////
// Model 3:


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
