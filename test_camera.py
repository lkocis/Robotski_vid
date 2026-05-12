from primesense import openni2

openni2.initialize(r"C:\Program Files\OpenNI2\Redist") 
dev = openni2.Device.open_any()
print(f"Success! Connected to: {dev.get_device_info().name}")
openni2.unload()