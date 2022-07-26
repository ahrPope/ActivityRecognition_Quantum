import os
from qiskit import IBMQ

#IBMQ.save_account("bb11e48e4c3982f3ac9cdb36e4b782da1ca2d9e1578984ac336a3f12f99e70ab89264625aace7109e93d209812e728617640cb5078f34fbd1302a14996ba403d")
IBMQ.load_account()
provider = IBMQ.get_provider(hub="ibm-q-lantik", group="udeusto", project="project1")  # Substitute with your provider.


#provider.runtime.delete_program("ibmq-qsvm-p2dyg3ZJwR")


program_json = os.path.join(os.getcwd(), "D:\Github\DL_HAR\quantum\Pegasos-QSVM.json")
program_data = os.path.join(os.getcwd(), "D:\Github\DL_HAR\quantum\Pegasos-QSVM.py")

program_id = provider.runtime.upload_program(
    data=program_data,
    metadata=program_json
)

#provider.runtime.set_program_visibility(program_id="ibmq-qsvm", public=True)

print(program_id)