#include<cassandra.h>
#include<iostream>

#include "batchpatchhandler.hpp"
#include "cass_pass.hpp" // credentials
extern Credentials* cred;

int main(){
  auto h = new BatchPatchHandler
    (2, NULL, "promort.data", "label", "data", "patch_id",
     cred->username, cred->password, {"cassandra_db"});
  for(int b=0; b<1000; ++b){
    vector<string> v({"92cd1dee-78da-4310-a35f-193a6d49809e",
	  "e4dd636b-cdbd-4ae2-a4d0-1e428da1a002",
	  "7752a14c-39bc-4a5e-9fc5-5b972dcb0764",
	  "0c23fc7e-aa31-4f6e-8a4d-c6e0267ab56c",
	  "2e5f5640-0e99-4052-9d0b-d4eb90f17858",
	  "c48ea534-1551-41ea-b0af-0a0edb274699",
	  "ecad19ee-cc8e-43b2-bef1-5cd81cc5debe",
	  "73de0ccf-4f64-4ca5-8143-595d4b2bba33",
	  "7309f746-9915-4dbe-a1e5-3706b4eb38d3",
	  "4666dc54-ab1b-4215-8db6-6d1f4f582a8d",
	  "7c388c11-218f-42a5-bf19-56170e1691a6",
	  "a1743593-127c-4443-afb0-59b224b4a2e1",
	  "3a5a91a0-0cf2-44e2-b369-6ed70ffa97e8",
	  "4253d64a-1cd5-4d95-9afb-c334cdd93d00",
	  "c968f781-7188-437b-914e-456ceda6ac81",
	  "e19aab82-8f51-4e12-a62f-c6132693d446",
	  "63f33697-2d73-4885-af12-6deb079de532",
	  "1b819af2-520f-4877-aada-ab04dd81ef38",
	  "b876c1aa-1c4e-4474-a3ca-7d120c4dcfa6",
	  "c903b789-9abc-494d-b310-7ab4d682c055",
	  "dfcc2ec4-e714-42fb-a6f6-d746482f97e4",
	  "40283418-a3be-4e95-9d36-3889f2dd1ea3",
	  "c98708c9-75b6-4b44-b134-c8ae5ed09a8b",
	  "a69e74fc-0034-4a2d-a03f-84c6a3d31aa3",
	  "c1ea4edd-6a7b-424a-8bb1-22baff9a3223",
	  "67e9912f-52de-4885-8542-d5ce5db85fa8",
	  "0a6c58ee-096a-4530-a682-ca48ed4eb585",
	  "407b5c03-9a52-4973-90fd-c2b2f539696d",
	  "b4361427-bee5-4a8a-bb87-30497f08f67e",
	  "3456aaff-aa70-49cf-af7c-411f7ba7e9a7",
	  "4b675d87-d428-4087-ac85-c23b335d8c4c",
	  "2940cbad-ced2-4d41-a2ac-b693284c5db1"
	  });
    auto z = h->load_batch(v, 0);
    cout << b << endl;
  }
}

