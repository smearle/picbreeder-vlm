/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package client.utilities;

/**
 * An ArgumentParser takes in the array of arguments for a command line application
 *   and returns the correct value for the option given.  If no option is given or
 *   if an option does not have a value, then an InvalidArgumentException is thrown.
 *    
 * @author Adam Campbell
 */
public class ArgumentParser {

	private String[] arguments;
	
	public ArgumentParser(String[] a){
		arguments = a;
	}
	
	/**
	 * Determines if an option exists in the arguments.
	 * 
	 * @param option The option
	 * @return <code>true</code> if it exists, <code>false</code> otherwise.
	 */
	public boolean hasOption(String option) {
		for(String s : arguments)
			if(s.equals(option))
				return true;
		
		return false;
	}
	
	/**
	 * The list of arguments given from a program should be
	 *   [option1] [value1] ... [optionN] [valueN]
	 * 
	 * @param option Command line options for a program.
	 * @return The value for the option.
	 */
	public String findArgument(String option){
		for(int i = 0; i < arguments.length; i++){
			if(arguments[i].equals(option)){
				if(i == arguments.length-1){
					throw new RuntimeException("Option \"" + option + "\" appeared at end of argument list and does not have a value.");
				}else{
					return arguments[i+1];
				}
			}
		}
		
		// option was not found
		throw new RuntimeException("Option \"" + option + "\" was not found in list.");
	}
}
