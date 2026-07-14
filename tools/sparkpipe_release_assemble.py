#!/usr/bin/env python3

import argparse
import datetime
import hashlib
import json
import os
import shutil


DIAGNOSTIC_ENVIRONMENT_BY_ROLE = {
    "spark0_gateway": {
        "SPARKPIPE_PP13_TRACE": "1",
    },
    "pp13_cuda_residentd": {
        "SPARKPIPE_STAGE_COMPLETION_DEBUG": "1",
        "SPARKPIPE_STAGE_PHASE_HASH": "1",
        "SPARKPIPE_HIDDEN_DUMP_DIR": "{state_root}/hidden_dumps",
    },
    "pp13_rank_daemon": {
        "SPARKPIPE_STAGE_COMPLETION_DEBUG": "1",
        "SPARKPIPE_PP13_TRACE": "1",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path,"rb") as source:
        while True:
            data = source.read(1024 * 1024)
            if not data:
                return digest.hexdigest()
            digest.update(data)


def set_role_max_active(manifest,max_active):
    manifest["max_active_sequence_count"] = max_active
    for role in manifest["roles"]:
        arguments = role.get("argv",[])
        for index in range(len(arguments) - 1):
            if arguments[index] == "--max-active":
                arguments[index + 1] = str(max_active)


def set_role_argument(manifest,role_name,argument,value):
    matching_roles = [role for role in manifest["roles"] if role.get("name") == role_name]
    if len(matching_roles) != 1:
        raise SystemExit("release must contain exactly one role named " + role_name)
    arguments = matching_roles[0].setdefault("argv",[])
    matches = [index for index,item in enumerate(arguments) if item == argument]
    if len(matches) > 1:
        raise SystemExit("role argument occurs more than once: " + argument)
    if matches:
        index = matches[0]
        if index + 1 >= len(arguments):
            raise SystemExit("role argument is missing its value: " + argument)
        arguments[index + 1] = str(value)
    else:
        arguments.extend([argument,str(value)])


def set_role_switch(manifest,role_name,argument,enabled):
    matching_roles = [role for role in manifest["roles"] if role.get("name") == role_name]
    if len(matching_roles) != 1:
        raise SystemExit("release must contain exactly one role named " + role_name)
    arguments = matching_roles[0].setdefault("argv",[])
    matches = [index for index,item in enumerate(arguments) if item == argument]
    if len(matches) > 1:
        raise SystemExit("role switch occurs more than once: " + argument)
    if matches:
        arguments.pop(matches[0])
    if enabled:
        arguments.append(argument)


def set_role_release_identity(manifest):
    values = {
        "SPARKPIPE_RELEASE_ID": manifest["release_id"],
        "SPARKPIPE_RELEASE_GIT_COMMIT": manifest["git_commit"],
        "SPARKPIPE_RELEASE_GENERATION": str(manifest["generation"]),
    }
    for role in manifest["roles"]:
        environment = role.setdefault("env",[])
        environment = [entry for entry in environment
                       if entry.split("=",1)[0] not in values]
        environment.extend(key + "=" + value for key,value in values.items())
        role["env"] = environment


def set_runtime_diagnostics(manifest,enabled):
    for role_name,values in DIAGNOSTIC_ENVIRONMENT_BY_ROLE.items():
        matching_roles = [role for role in manifest["roles"]
                          if role.get("name") == role_name]
        if len(matching_roles) != 1:
            raise SystemExit(
                "release must contain exactly one role named " + role_name)
        role = matching_roles[0]
        environment = role.get("env",[])
        role["env"] = [entry for entry in environment
                       if entry.split("=",1)[0] not in values]
        if enabled:
            role["env"].extend(key + "=" + value
                               for key,value in values.items())


def write_manifest(root,manifest):
    for entry in manifest["files"]:
        path = os.path.join(root,entry["path"])
        if not os.path.isfile(path):
            raise SystemExit("missing release file: " + entry["path"])
        entry["bytes"] = os.path.getsize(path)
        entry["sha256"] = sha256(path)
    path = os.path.join(root,"sparkpipe.json")
    with open(path,"w",encoding="utf-8") as target:
        json.dump(manifest,target,indent=2,sort_keys=True)
        target.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--release-id",required=True)
    parser.add_argument("--git-commit",required=True)
    parser.add_argument("--max-active",type=int)
    parser.add_argument("--kv-pool-tokens",type=int)
    parser.add_argument("--kv-logical-blocks",type=int,required=True)
    parser.add_argument("--mtp",action="store_true")
    parser.add_argument("--without-diagnostics",action="store_true")
    parser.add_argument("--replace",action="append",default=[])
    arguments = parser.parse_args()
    temporary = arguments.output + ".assembling." + str(os.getpid())
    if os.path.exists(arguments.output) or os.path.exists(temporary):
        raise SystemExit("release output already exists")
    shutil.copytree(arguments.template,temporary)
    manifest_path = os.path.join(temporary,"sparkpipe.json")
    with open(manifest_path,"r",encoding="utf-8") as source:
        manifest = json.load(source)
    allowed = {entry["path"] for entry in manifest["files"]}
    for replacement in arguments.replace:
        relative,source = replacement.split("=",1)
        if relative not in allowed:
            raise SystemExit("replacement is not in manifest: " + relative)
        if not os.path.isfile(source):
            raise SystemExit("missing replacement: " + source)
        shutil.copy2(source,os.path.join(temporary,relative))
    manifest["release_id"] = arguments.release_id
    manifest["git_commit"] = arguments.git_commit
    manifest["generation"] = int(datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S"))
    set_role_release_identity(manifest)
    if arguments.max_active is not None:
        if arguments.max_active < 1 or arguments.max_active > 1024:
            raise SystemExit("max-active must be in 1..1024")
        set_role_max_active(manifest,arguments.max_active)
    if arguments.kv_pool_tokens is not None:
        if arguments.kv_pool_tokens < 1:
            raise SystemExit("kv-pool-tokens must be positive")
        set_role_argument(
            manifest,"pp13_cuda_residentd","--kv-pool-tokens",
            arguments.kv_pool_tokens)
    if arguments.kv_logical_blocks < 1:
        raise SystemExit("kv-logical-blocks must be positive")
    set_role_argument(
        manifest,"spark0_gateway","--kv-logical-blocks",
        arguments.kv_logical_blocks)
    set_role_switch(manifest,"spark0_gateway","--mtp",arguments.mtp)
    set_role_switch(manifest,"pp13_cuda_residentd","--mtp",arguments.mtp)
    set_runtime_diagnostics(manifest,not arguments.without_diagnostics)
    write_manifest(temporary,manifest)
    os.rename(temporary,arguments.output)


if __name__ == "__main__":
    main()
