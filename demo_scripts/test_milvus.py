from pymilvus import connections

# Connect to Milvus server
try:
    connections.connect(
        alias="default", 
        host="localhost",
        port="19530"
    )
    print("Successfully connected to Milvus")
except Exception as e:
    print(f"Failed to connect to Milvus: {e}")

# Test basic operations
try:
    from pymilvus import utility
    if utility.get_server_version():
        print("Milvus server version:", utility.get_server_version())
    else:
        print("Could not get Milvus version")
except Exception as e:
    print(f"Error testing Milvus: {e}")

# Close connection
connections.disconnect("default")
