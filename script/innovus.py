import os
import sys
import argparse

def parseArgs(): 
    parser = argparse.ArgumentParser()
                        
    parser.add_argument("-o", "--output", required=False, 
                        action="store", \
                        default="result")
    parser.add_argument("--timeout", required=False, 
                        action="store", \
                        default=None)
    
    #=======================================
    # MMMC
    #=======================================
    parser.add_argument("--typical", required=False, 
                        action="append", \
                        default=[])
    parser.add_argument("--best", required=False, 
                        action="append", \
                        default=[])
    parser.add_argument("--worst", required=False, 
                        action="append", \
                        default=[])
    parser.add_argument("--cap", required=False, 
                        action="append", \
                        default=[])
    parser.add_argument("--synthed_sdc", required=True, 
                        action="append", \
                        default=[])
    
    #=======================================
    # Init
    #=======================================
    parser.add_argument("--lef", required=False, 
                        action="append", \
                        default=[])
    parser.add_argument("--synthed_hdl", required=True, 
                        action="append", \
                        default=[])
    parser.add_argument("--mmmc", required=False, 
                        action="append", \
                        default=[])
    
    #=======================================
    # Floorplan
    #=======================================
    parser.add_argument("--margin_by", required=False, 
                        action="store", default="io") # {io, die}
    parser.add_argument("--origin", required=False, 
                        action="store", default="llcorner") # {llcorner, center}
    parser.add_argument("--mode", required=False, 
                        action="store", default="r") # {r, su}
    parser.add_argument("--aspect", required=False, 
                        action="store", default=1.0)
    parser.add_argument("--density", required=False, 
                        action="store", default=0.7)
    parser.add_argument("--margin", required=False, 
                        action="store", default=2.6)
    parser.add_argument("--make_path_groups", required=False, 
                        action="store", default="true") # {false, true}
    
    
    #=======================================
    # Place
    #=======================================
    # setPlaceMode
    # -place_detail_irdrop_aware_effort
    parser.add_argument("--detail_irdrop_aware", required=False, 
                        action="store", default="none") # {none, low, medium, high}
    # -place_detail_irdrop_aware_timing_effort
    parser.add_argument("--detail_irdrop_aware_timing", required=False, 
                        action="store", default="none") # {none, standard, high}
    # -place_detail_wire_length_opt_effort
    parser.add_argument("--detail_wire_length_opt", required=False, 
                        action="store", default="medium") # {none, medium, high}
    # -place_global_activity_power_driven, {false, true}, default: false
    # -place_global_activity_power_driven_effort, {none, standard, high}, default: standard
    parser.add_argument("--global_activity_power_driven", required=False, 
                        action="store", default="none") # {none, standard, high}
    # -place_global_align_macro
    parser.add_argument("--global_align_macro", required=False, 
                        action="store", default="false") # {false, true}
    # -place_global_clock_gate_aware
    parser.add_argument("--global_clock_gate_aware", required=False, 
                        action="store", default="true") # {false, true}
    # -place_global_clock_power_driven, {false, true}, default: true
    # -place_global_clock_power_driven_effort, {low, standard, high}, default: low
    parser.add_argument("--global_clock_power_driven_effort", required=False, 
                        action="store", default="standard") # {none, low, standard, high} 
                        # NOTE: in Innovus 17, there are only {none, standard, high}
    # -place_global_cong_effort
    parser.add_argument("--global_cong_effort", required=False, 
                        action="store", default="auto") # {low, medium, high, auto}
    # -place_global_place_io_pins
    parser.add_argument("--global_place_io_pins", required=False, 
                        action="store", default="false") # {false, true}
    # -place_global_soft_guide_strength
    parser.add_argument("--global_soft_guide_strength", required=False, 
                        action="store", default="low") # {low, medium, high}
    # -place_global_timing_effort
    parser.add_argument("--global_timing_effort", required=False, 
                        action="store", default="medium") # {medium, high}
    # -place_global_uniform_density
    parser.add_argument("--global_uniform_density", required=False, 
                        action="store", default="false") # {false, true}
    # placeDesign
    # -noPrePlaceOpt
    parser.add_argument("--pre_place_opt", required=False, 
                        action="store", default="false") # {false, true}
    
    #=======================================
    # Route
    #=======================================
    # setNanoRouteMode
    # -droutePostRouteViaPillarEffort
    parser.add_argument("--post_via_pillar_effort", required=False, 
                        action="store", default="low") # {none, low, medium, high}
    # -drouteUseMultiCutViaEffort
    parser.add_argument("--post_multicut_via_effort", required=False, 
                        action="store", default="low") # {low, medium, high}
    # -routeWithLithoDriven
    parser.add_argument("--litho_driven", required=False, 
                        action="store", default="false") # {false, true}
    # -routeWithSiDriven
    parser.add_argument("--si_driven", required=False, 
                        action="store", default="true") # {false, true}
    # -routeWithTimingDriven
    parser.add_argument("--timing_driven", required=False, 
                        action="store", default="true") # {false, true}
    # routeDesign
    # viaOpt
    parser.add_argument("--via_opt", required=False, 
                        action="store", default="false") # {false, true}
    # viaPillarOpt
    parser.add_argument("--via_pillar_opt", required=False, 
                        action="store", default="false") # {false, true}
    # wireOpt
    parser.add_argument("--wire_opt", required=False, 
                        action="store", default="false") # {false, true}
    # optDesign
    # -hold
    parser.add_argument("--hold", required=False, 
                        action="store", default="false") # {false, true}
    
    #=======================================
    # Report
    #=======================================
    parser.add_argument("--timing", required=False, 
                        action="store", \
                        default="")
    parser.add_argument("--area", required=False, 
                        action="store", \
                        default="")
    parser.add_argument("--power", required=False, 
                        action="store", \
                        default="")
    parser.add_argument("--drc", required=False, 
                        action="store", \
                        default="")
    
    return parser.parse_args()


def createMMMC(args, outfile): 
    if len(args.typical) == 0: 
        args.typical = ["lib/tcbn65lptc.lib"]
    if len(args.best) == 0: 
        args.best = ["lib/tcbn65lpbc.lib"]
    if len(args.worst) == 0: 
        args.worst = ["lib/tcbn65lpwc.lib"]
    if len(args.cap) == 0: 
        args.cap = ["lib/cln65lp_1p06m+alrdl_top1_typical.captable"]
    
    typical = ""
    for name in args.typical: 
        typical += name + " "
    best = ""
    for name in args.best: 
        best += name + " "
    worst = ""
    for name in args.worst: 
        worst += name + " "
    captable = ""
    for name in args.cap: 
        captable += name + " "
    sdc = ""
    for name in args.synthed_sdc: 
        sdc += name + " "
    
    info = '''
if {![namespace exists ::IMEX]} { namespace eval ::IMEX {} }
set ::IMEX::dataVar [file dirname [file normalize [info script]]]

create_library_set -name libs_typical\\
   -timing\\
    [list %s]
create_library_set -name libs_bc\\
   -timing\\
    [list %s]
create_library_set -name libs_wc\\
   -timing\\
    [list %s]
create_rc_corner -name typical\\
   -cap_table %s\\
   -preRoute_res 1\\
   -postRoute_res 1\\
   -preRoute_cap 1\\
   -postRoute_cap 1\\
   -postRoute_xcap 1\\
   -preRoute_clkres 0\\
   -preRoute_clkcap 0\\
   -T 25
create_delay_corner -name delay_default\\
   -rc_corner typical\\
   -early_library_set libs_bc\\
   -late_library_set libs_typical
create_constraint_mode -name constraints_default\\
   -sdc_files\\
    [list %s]
create_analysis_view -name analysis_default -constraint_mode constraints_default -delay_corner delay_default
set_analysis_view -setup [list analysis_default] -hold [list analysis_default]
    ''' % (typical, best, worst, captable, sdc)
    with open(outfile, "w") as fout: 
        fout.write(info)


def createInit(args, outfile): 
    if len(args.lef) == 0: 
        args.lef = ["lib/tcbn65lp_6lmT1.lef"]
    if len(args.mmmc) == 0: 
        args.mmmc = ["/".join(outfile.split("/")[:-1] + ["mmmc.tcl"])]
    with open(outfile, "w") as fout: 
        info = "{ "
        for name in args.lef: 
            info += name + " "
        info += "}"
        fout.write("set init_lef_file " + info + "\n")
        
        fout.write("set init_design_settop 0\n")
        
        info = "{ "
        for name in args.synthed_hdl: 
            info += name + " "
        info += "}"
        fout.write("set init_verilog " + info + "\n")
        
        info = "{ "
        for name in args.mmmc: 
            info += name + " "
        info += "}"
        fout.write("set init_mmmc_file " + info + "\n")
        
        fout.write("set init_design_uniquify 1\n")
        fout.write("init_design\n")


def createFloorplan(args, outfile): 
    option = "-coreMarginsBy %s -fplanOrigin %s -%s" % \
             (args.margin_by, args.origin, args.mode)
             
    pinAssign = '''
# Take all ports and split into halves

set all_ports       [dbGet top.terms.name -v *clk*]

set num_ports       [llength $all_ports]
set half_ports_idx  [expr $num_ports / 2]

set pins_left_half  [lrange $all_ports 0               [expr $half_ports_idx - 1]]
set pins_right_half [lrange $all_ports $half_ports_idx [expr $num_ports - 1]     ]

# Take all clock ports and place them center-left

set clock_ports     [dbGet top.terms.name *clk*]
set half_left_idx   [expr [llength $pins_left_half] / 2]

if { $clock_ports != 0 } {
  for {set i 0} {$i < [llength $clock_ports]} {incr i} {
    set pins_left_half \
      [linsert $pins_left_half $half_left_idx [lindex $clock_ports $i]]
  }
}

# Spread the pins evenly across the left and right sides of the block

set ports_layer M4

editPin -layer $ports_layer -pin $pins_left_half  -side LEFT  -spreadType SIDE
editPin -layer $ports_layer -pin $pins_right_half -side RIGHT -spreadType SIDE
    '''
    
    makePathGroups = '''
# Reset all existing path groups, including basic path groups

reset_path_group -all

# Reset all options set on all path groups

resetPathGroupOptions

# Create collection for each category

set inputs   [all_inputs -no_clocks]
set outputs  [all_outputs]
set icgs     [filter_collection [all_registers] "is_integrated_clock_gating_cell == true"]
set regs     [remove_from_collection [all_registers -edge_triggered] $icgs]
set allregs  [all_registers]

# Create collection for all macros

set blocks      [ dbGet top.insts.cell.baseClass block -p2 ]
set macro_refs  [ list ]
set macros      [ list ]

# If the list of blocks is non-empty, filter out non-physical blocks

set blocks_exist  [ expr [ lindex $blocks 0 ] != 0 ]

if { $blocks_exist } {
  foreach b $blocks {
    set cell    [ dbGet $b.cell ]
    set isBlock [ dbIsCellBlock $cell ]
    set isPhys  [ dbGet $b.isPhysOnly ]
    # Return all blocks that are _not_ physical-only (e.g., filter out IO bondpads)
    if { [ expr $isBlock && ! $isPhys ] } {
      puts [ dbGet $b.name ]
      lappend macro_refs $b
      lappend macros     [ dbGet $b.name ]
    }
  }
}

# Group paths (for any groups that exist)

group_path -name In2Out -from $inputs -to $outputs

if { $allregs != "" } {
  group_path -name In2Reg  -from $inputs  -to $allregs
  group_path -name Reg2Out -from $allregs -to $outputs
}

if { $regs != "" } {
  group_path -name Reg2Reg -from $regs -to $regs
}

if { $allregs != "" && $icgs != "" } {
  group_path -name Reg2ClkGate -from $allregs -to $icgs
}

if { $macros != "" } {
  group_path -name All2Macro -to   $macros
  group_path -name Macro2All -from $macros
}

# High-effort path groups

if { $macros != "" } {
  setPathGroupOptions All2Macro -effortLevel high
  setPathGroupOptions Macro2All -effortLevel high
}

if { $regs != "" } {
  setPathGroupOptions Reg2Reg -effortLevel high
}
'''
    
    with open(outfile, "w") as fout: 
        fout.write("# Floorplaning \n\n")
        fout.write("floorPlan %s %f %f %f %f %f %f" % \
                   (option, \
                    float(args.aspect), float(args.density), float(args.margin), \
                    float(args.margin), float(args.margin), float(args.margin)))
                    
        fout.write("\n\n# Pin assignment \n\n")
        fout.write(pinAssign)
        
        if args.make_path_groups: 
            fout.write("\n\n# Make path groups \n\n")
            fout.write(makePathGroups)


def createPlace(args, outfile): 
    
    setPlaceMode = "setPlaceMode"
    # setPlaceMode += " -place_detail_irdrop_aware_effort " + args.detail_irdrop_aware
    # setPlaceMode += " -place_detail_irdrop_aware_timing_effort " + args.detail_irdrop_aware_timing
    setPlaceMode += " -place_detail_wire_length_opt_effort " + args.detail_wire_length_opt
    # setPlaceMode += " -place_global_activity_power_driven " + ("false" if args.global_activity_power_driven == "none" else "true")
    # setPlaceMode += " -place_global_activity_power_driven_effort " + args.global_activity_power_driven
    # setPlaceMode += " -place_global_align_macro " + args.global_align_macro
    setPlaceMode += " -place_global_clock_gate_aware " + args.global_clock_gate_aware
    setPlaceMode += " -place_global_clock_power_driven " + ("false" if args.global_clock_power_driven_effort == "none" else "true")
    setPlaceMode += " -place_global_clock_power_driven_effort " + ("standard" if args.global_clock_power_driven_effort == "none" else args.global_clock_power_driven_effort)
    setPlaceMode += " -place_global_cong_effort " + args.global_cong_effort
    setPlaceMode += " -place_global_place_io_pins " + args.global_place_io_pins
    setPlaceMode += " -place_global_soft_guide_strength " + args.global_soft_guide_strength
    setPlaceMode += " -place_global_timing_effort " + args.global_timing_effort
    setPlaceMode += " -place_global_uniform_density " + args.global_uniform_density
    
    placeDesign = "placeDesign " + ("" if args.pre_place_opt == "true" else "-noPrePlaceOpt")
    
    with open(outfile, "w") as fout: 
        fout.write(setPlaceMode + "\n")
        fout.write(placeDesign + "\n")
        fout.write("optDesign -preCTS\n")
        fout.write("place_opt_design\n")
        fout.write("create_ccopt_clock_tree_spec\n")
        fout.write("ccopt_design\n")
        fout.write("optDesign -postCTS\n")


def createRoute(args, outfile): 
    
    setNanoRouteMode = "setNanoRouteMode"
    # setNanoRouteMode += " -droutePostRouteViaPillarEffort " + args.post_via_pillar_effort # INNOVUS 17.1 does not support it
    setNanoRouteMode += " -drouteUseMultiCutViaEffort " + args.post_multicut_via_effort
    setNanoRouteMode += " -routeWithLithoDriven " + args.litho_driven
    setNanoRouteMode += " -routeWithSiDriven " + args.si_driven
    setNanoRouteMode += " -routeWithTimingDriven " + args.timing_driven
    
    routeDesign = "routeDesign -globalDetail"
    routeDesign += "\nrouteDesign -viaOpt" if args.via_opt else ""
    # routeDesign += "\nrouteDesign -viaPillarOpt" if args.via_pillar_opt else ""
    routeDesign += "\nrouteDesign -wireOpt" if args.wire_opt else ""
    
    optDesign = "optDesign -postRoute"
    optDesign += " -hold" if args.hold == "true" else ""
    
    with open(outfile, "w") as fout: 
        fout.write(setNanoRouteMode + "\n")
        fout.write(routeDesign + "\n")
        fout.write("setAnalysisMode -analysisType onChipVariation\n")
        fout.write(optDesign + "\n")
        fout.write("timeDesign -postRoute\n")


def createReport(args, outfile): 
    with open(outfile, "w") as fout: 
        fout.write("report_timing > %s\n" % args.timing)
        fout.write("report_area > %s\n" % args.area)
        fout.write("report_power > %s\n" % args.power)
        fout.write("verify_drc -report %s\n" % args.drc)


def createScripts(args): 
    basedir = args.output
    fileMMMC = basedir + "/script/mmmc.tcl"
    fileInit = basedir + "/script/init.tcl"
    fileFloorplan = basedir + "/script/floorplan.tcl"
    filePlace = basedir + "/script/place.tcl"
    fileRoute = basedir + "/script/route.tcl"
    fileReport = basedir + "/script/report.tcl"
    createMMMC(args, fileMMMC)
    createInit(args, fileInit)
    createFloorplan(args, fileFloorplan)
    createPlace(args, filePlace)
    createRoute(args, fileRoute)
    createReport(args, fileReport)
    
    with open(basedir + "/script/innovus.tcl", "w") as fout: 
        fout.write("source " + fileInit + "\n")
        fout.write("source " + fileFloorplan + "\n")
        fout.write("source " + filePlace + "\n")
        fout.write("source " + fileRoute + "\n")
        fout.write("source " + fileReport + "\n")


def run(basedir, timeout=None): 
    prefix = "timeout %d " % (int(timeout), ) if not timeout is None else ""
    os.system(prefix + "/opt2/cadence/INNOVUS171/bin/innovus -no_gui -no_logv -log %s -batch -execute 'source %s' > %s 2>&1" % \
              (basedir + "/log/innovus", basedir + "/script/innovus.tcl", "/dev/null"))


def result(timingfile, powerfile, areafile): 
    slack = None
    with open(timingfile, "r") as fin: 
        for line in fin.readlines(): 
            splited = line.split()
            if len(splited) > 3 and splited[0] == "=" and splited[1] == "Slack" and splited[2] == "Time": 
                slack = float(splited[3])
        
    power = None # NOTE: get the total power
    with open(powerfile, "r") as fin: 
        for line in fin.readlines(): 
            splited = line.split()
            if len(splited) > 2 and splited[0] == "Total" and splited[1] == "Power:": 
                power = float(splited[2])
        
    area = None
    with open(areafile, "r") as fin: 
        for line in fin.readlines(): 
            splited = line.split()
            # Innovus 17.1
            if len(splited) > 3 and splited[0] == "0": 
                area = float(splited[3])
            # Innovus 20.1
            # if len(splited) > 2 and splited[1].isdigit(): 
            #     area = float(splited[2])
            #     break
        
    return slack, power, area
    

def verify(drcfile): 
    result = False
    with open(drcfile, "r") as fin: 
        for line in fin.readlines(): 
            if line.strip() == "No DRC violations were found": 
                result = True
    return result

def main(args): 
    
    basedir = args.output
    if len(args.timing) == 0: 
        args.timing = basedir + "/log/timing_innovus.log"
    if len(args.area) == 0: 
        args.area = basedir + "/log/area_innovus.log"
    if len(args.power) == 0: 
        args.power = basedir + "/log/power_innovus.log"
    if len(args.drc) == 0: 
        args.drc = basedir + "/log/drc_innovus.log"
    if len(args.synthed_hdl) == 0: 
        args.synthed_hdl = [basedir + "/script/synthed.v"]
    if len(args.synthed_sdc) == 0: 
        args.synthed_sdc = [basedir + "/script/synthed.sdc"]
    
    if not os.path.exists(basedir): 
        os.mkdir(basedir)
    if not os.path.exists(basedir + "/script"): 
        os.mkdir(basedir + "/script")
    if not os.path.exists(basedir + "/log"): 
        os.mkdir(basedir + "/log")
        
    createScripts(args)
    run(basedir, args.timeout)
    valid = verify(args.drc)
    res = ["ERR", "ERR", "ERR", ]
    if valid: 
        res = result(args.timing, args.power, args.area)
    output = str(res[0]) + " " + str(res[1]) + " " + str(res[2])
    with open(basedir + "/log/innovus_result.log", "w") as fout: 
        fout.write(output)
    print(output)

    # os.system("rm -rf " + basedir + "/log/innovus.*")


if __name__ == "__main__": 
    main(parseArgs())
