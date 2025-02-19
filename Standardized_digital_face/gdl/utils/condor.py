import os
import subprocess
import stat


# def execute_on_cluster(config_fname):
def execute_on_cluster(cluster_script_path, args, submission_dir_local_mount,
                       submission_dir_cluster_side=None,
                       cluster_repo_dir='/home/ranecek/workspace/repos/gdl',
                       cpus=1, gpus=0, mem_gb=4, num_jobs=1, bid=10, max_time_h=None,
                       max_price=5000,
                       job_name="skynet",
                       python_bin='python',
                       #env='work36',
                       env='work36_cu11',
                       username='rdanecek',
                       gpu_mem_requirement_mb=None,
                       gpu_mem_requirement_mb_max=None,
                       cuda_capability_requirement=None,
                       max_concurrent_jobs=None,
                       concurrency_tag = None,
                       modules_to_load = None,
                       chmod=True):
    modules_to_load = modules_to_load or []
    submission_dir_cluster_side = submission_dir_cluster_side or submission_dir_local_mount
    logdir = 'logs'

    st = script_template
    # st = st.replace('<<CONFIG_FNAME>>', config_fname)
    st = st.replace('<<REPO_ROOT>>', cluster_repo_dir)
    st = st.replace('<<PYTHON_BIN>>', python_bin)
    st = st.replace('<<SCRIPT_NAME>>', cluster_script_path)
    st = st.replace('<<ENV>>', env)
    modules = ""
    if len(modules_to_load) > 0:
        modules = f"module load {' '.join(modules_to_load)}"
    st = st.replace('<<MODULES>>', modules)
    script_fname = os.path.join(submission_dir_local_mount, 'run.sh')


    cs = condor_template
    cs = cs.replace('<<PATH>>', logdir)
    cs = cs.replace('<<ARGS>>', args)
    cs = cs.replace('<<SCRIPTNAME>>', os.path.basename(script_fname))
    cs = cs.replace('<<JOBNAME>>', job_name)
    cs = cs.replace('<<CPU_COUNT>>', str(int(cpus)))
    cs = cs.replace('<<GPU_COUNT>>', str(int(gpus)))
    cs = cs.replace('<<MEMORYMBS>>', str(int(mem_gb * 1024)))
    cs = cs.replace('<<MAX_TIME>>', str(int(max_time_h * 3600)))
    cs = cs.replace('<<MAX_PRICE>>', str(int(max_price)))
    cs = cs.replace('<<NJOBS>>', str(num_jobs))


    if num_jobs>1:
        cs = cs.replace('<<PROCESS_ID>>', ".$(Process)")
    else:
        cs = cs.replace('<<PROCESS_ID>>', "")

    requirements = []

    if cuda_capability_requirement:
        if isinstance(cuda_capability_requirement, int):
            cuda_capability_requirement = str(cuda_capability_requirement) + ".0"
        requirements += [f"( TARGET.CUDACapability>={cuda_capability_requirement} )"]
    if gpu_mem_requirement_mb:
        requirements += [f"( TARGET.CUDAGlobalMemoryMb>={gpu_mem_requirement_mb} )"]
    if gpu_mem_requirement_mb_max:
        requirements += [f"( TARGET.CUDAGlobalMemoryMb<={gpu_mem_requirement_mb_max} )"]
    if len(requirements) > 0:
        requirements = " && ".join(requirements)
        requirements = "requirements=" + requirements

    cs = cs.replace('<<REQUIREMENTS>>', requirements)
    condor_fname = os.path.join(submission_dir_local_mount, 'run.condor')

    concurrency_string = ""
    if concurrency_tag is not None and max_concurrent_jobs is not None:
        concurrency_limits = 10000 // max_concurrent_jobs
        concurrency_string += f"concurrency_limits = user.{concurrency_tag}:{concurrency_limits}"
    cs = cs.replace('<<CONCURRENCY>>', concurrency_string)

    # write files
    with open(script_fname, 'w') as fp:
        fp.write(st)
    os.chmod(script_fname, stat.S_IXOTH | stat.S_IWOTH | stat.S_IREAD | stat.S_IEXEC | stat.S_IXUSR | stat.S_IRUSR)  # make executable
    with open(condor_fname, 'w') as fp:
        fp.write(cs)
    os.chmod(condor_fname, stat.S_IXOTH | stat.S_IWOTH | stat.S_IREAD | stat.S_IEXEC | stat.S_IXUSR | stat.S_IRUSR)  # make executable

    if chmod:
        cmd = f'cd {submission_dir_cluster_side} && ' \
              f'mkdir {logdir} && ' \
              f'chmod +x {os.path.basename(script_fname)} && ' \
              f'chmod +x {os.path.basename(condor_fname)} && ' \
              f'condor_submit_bid {bid} {os.path.basename(condor_fname)}'
    else:
        cmd = f'cd {submission_dir_cluster_side} && ' \
              f'mkdir {logdir} && ' \
              f'condor_submit_bid {bid} {os.path.basename(condor_fname)}'

    print("Called the following on the cluster: ")
    print(cmd)
    # subprocess.call(["ssh", "%s@login.cluster.is.localnet" % (username,)] + [cmd])
    # subprocess.call(["ssh", "%s@login1.cluster.is.localnet" % (username,)] + [cmd])
    subprocess.call(["ssh", "%s@login2.cluster.is.localnet" % (username,)] + [cmd])
    print("Done")
