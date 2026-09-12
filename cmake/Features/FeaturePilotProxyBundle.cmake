# Export development/test bundles; deployments supply a pinned, calibrated bundle. OFF skips export,
# ON requires the CLI, and AUTO exports when the CLI is present. Re-export on each build so changes
# to the installed package are included.

if(NOT "${PILOTPROXY_EXPORT_BUNDLE}" STREQUAL "OFF")
    find_program(PILOTPROXY_CLI pilot-proxy)
    if(NOT PILOTPROXY_CLI)
        if("${PILOTPROXY_EXPORT_BUNDLE}" STREQUAL "ON")
            message(
                FATAL_ERROR
                    "PILOTPROXY_EXPORT_BUNDLE=ON but the pilot-proxy CLI was not found on PATH. "
                    "Install it (pip install from https://github.com/WVURAIL/pilot-proxy) or set "
                    "PILOTPROXY_EXPORT_BUNDLE=AUTO/OFF.")
        else()
            kmsg_status("PilotProxy bundle export skipped (AUTO: pilot-proxy CLI not found).")
        endif()
    else()
        set(PILOTPROXY_BUNDLE_DIR
            "${CMAKE_BINARY_DIR}/pilotproxy_bundle"
            CACHE PATH "Output directory for the exported PilotProxy runtime weight bundle")
        set(PILOTPROXY_CHANNEL_RANGE
            "14:36"
            CACHE STRING "ATSC physical-channel range for the exported PilotProxy bundle")
        # Explicit profiles skip discovery in the CLI's Python environment.
        if(NOT PILOTPROXY_RECEIVER_PROFILE OR NOT PILOTPROXY_DETECTOR_CORE_PROFILE)
            get_filename_component(_pilotproxy_bindir "${PILOTPROXY_CLI}" DIRECTORY)
            find_program(
                PILOTPROXY_PYTHON
                NAMES python3 python
                HINTS "${_pilotproxy_bindir}")
            execute_process(
                COMMAND "${PILOTPROXY_PYTHON}" -c
                        "from pilot_proxy.paths import CONFIGS_DIR; print(CONFIGS_DIR)"
                RESULT_VARIABLE _pilotproxy_paths_result
                OUTPUT_VARIABLE _pilotproxy_configs
                ERROR_VARIABLE _pilotproxy_paths_error
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(NOT "${_pilotproxy_paths_result}" STREQUAL "0")
                message(
                    FATAL_ERROR
                        "Could not locate PilotProxy profiles: ${_pilotproxy_paths_error}. "
                        "Set PILOTPROXY_PYTHON to the CLI's Python interpreter or provide "
                        "PILOTPROXY_RECEIVER_PROFILE and PILOTPROXY_DETECTOR_CORE_PROFILE.")
            endif()
        endif()
        set(PILOTPROXY_RECEIVER_PROFILE
            "${_pilotproxy_configs}/receiver_profiles/chord_dtv_fengine.json"
            CACHE FILEPATH "Receiver profile for the exported PilotProxy bundle")
        set(PILOTPROXY_DETECTOR_CORE_PROFILE
            "${_pilotproxy_configs}/detector_core/pilotproxy_cuda_local_reference_power_ratio.json"
            CACHE FILEPATH "Detector-core profile for the exported PilotProxy bundle")
        add_custom_target(
            pilotproxy-bundle ALL
            COMMAND
                ${PILOTPROXY_CLI} export-runtime-weight-bundle --receiver-profile
                ${PILOTPROXY_RECEIVER_PROFILE} --detector-core-profile
                ${PILOTPROXY_DETECTOR_CORE_PROFILE} --weight-coordinate-system
                post_spectral_sense_normalization --physical-channel-range
                ${PILOTPROXY_CHANNEL_RANGE} --output-dir ${PILOTPROXY_BUNDLE_DIR}
            COMMAND ${PILOTPROXY_CLI} validate-runtime-weight-bundle --bundle-dir
                    ${PILOTPROXY_BUNDLE_DIR}
            BYPRODUCTS ${PILOTPROXY_BUNDLE_DIR}/pilot_profiles.json
                       ${PILOTPROXY_BUNDLE_DIR}/weights.bin
            COMMENT "Exporting + validating the PilotProxy runtime weight bundle"
            VERBATIM)
        kmsg_status("PilotProxy bundle export enabled -> ${PILOTPROXY_BUNDLE_DIR}")
    endif()
endif()
