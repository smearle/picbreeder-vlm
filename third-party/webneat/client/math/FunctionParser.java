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

package client.math;

import java.util.*;

/**
 * The FunctionParser is responsible for taking the mathematical expressions
 * found in the genome activation functions and turning them into executable
 * code for the CPPN.
 * 
 * @author Nick Beato
 */
public class FunctionParser {
	private static final FunctionParser singleton = new FunctionParser();
	
	/**
	 * The representation of an identity function in text.
	 */
	public static final String IDENTITY = "identity(x)";
	
	private Map <String, Function> map;
	
	private FunctionParser() {
		map = new HashMap <String, Function> ();
		map.put("identity(x)", new Identity());
		
		// temporary for now so the reflected instances are added using sample tables
		addUnipolarFunction("sigmoid(x)", new Sigmoid(), true);
		addUnipolarFunction("gaussian(x)", new Gaussian(), true);
		//addUnipolarFunction("sigmoid(x)", new PowerSeriesSigmoid(), false);
		//addUnipolarFunction("gaussian(x)", new PowerSeriesGaussian(), false);
		addBipolarFunction("sin(x)", new Sine(), false);
		addBipolarFunction("cos(x)", new Cosine(), false);
		addUnipolarFunction("delphi.sigmoid(x)", new client.math.delphi.Sigmoid(), true);
		addUnipolarFunction("delphi.gaussian(x)", new client.math.delphi.Gaussian(), true);
	}
	
	private void addUnipolarFunction(String name, Function function, boolean createTable) {
		Function f = new UnipolarToBipolar(function);
		if(createTable)
			f = createSampledFunction(f);
		map.put(name, f);
	}
	
	private void addBipolarFunction(String name, Function function, boolean createTable) {
		Function f = function;
		if(createTable)
			f = createSampledFunction(f);
		map.put(name, f);
	}
	
	private Function createSampledFunction(Function f) {
		return new SampledBoundedFunction(f, 1e-3);
	}

	/**
	 * Parses the textual representation of the specified function
	 * and returns an invokable version of it. 
	 * 
	 * @param function The function text
	 * @return The executable function code
	 */
	public Function parse(String function) {        
		return map.get(function);
	} 

	/**
	 * Retrieves the parser.
	 * 
	 * @return The singleton function parser.
	 */
	public static FunctionParser instance() {        
		return singleton;
	} 
}
