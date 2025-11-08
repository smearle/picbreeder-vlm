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

package test;

import java.util.*;

import client.AbstractParameterTable;

public final class TestParameters extends AbstractParameterTable {
	final static int LOW_RES_STRIDE = 4; // must divide WIDTH and HEIGHT
	final static int WIDTH = 128;
	final static int HEIGHT = 128;
	final static int ROWS = 3;
	final static int COLUMNS = 5;
	
	public TestParameters() {
		addItemToSet("activations", "sigmoid(x)");
		addItemToSet("activations", "gaussian(x)");
		addItemToSet("activations", "sin(x)");
		
		addItemToSet("inputs", "x");
		addItemToSet("inputs", "y");
		addItemToSet("inputs", "d");
		addItemToSet("inputs", "bias");
		
		addItemToSet("outputs", "ink");

		addItemToSet("generators", "client.evolution.generators.MutateLinks", 2);
		addItemToSet("generators", "client.evolution.generators.AddAcyclicLink");
		addItemToSet("generators", "client.evolution.generators.AddNodes");
		
		// model the delphiNEAT for now
		setParameter("activation", "bias", "1.0");
		setParameter("activation", "x scale", Double.toString(1.0 * WIDTH / HEIGHT)); // for aspect ratio correction
		setParameter("activation", "y scale", "1.0");
		setParameter("activation", "distance scale", Double.toString(Math.sqrt(2.0)));
		
		setParameter("evolution", "weight mutation rate", "0.20");
		setParameter("evolution", "activation mutation rate", "0.05");
		setParameter("evolution", "max link weight", "3.0");
		setParameter("evolution", "add node rate", "0.07");
		setParameter("evolution", "add link rate", "0.10");
		setParameter("evolution", "hidden nodes", "1");
		setParameter("evolution", "population size", Integer.toString(ROWS * COLUMNS));
		
		setParameter("display", "rows", Integer.toString(ROWS));
		setParameter("display", "columns", Integer.toString(COLUMNS));
		setParameter("display", "width", Integer.toString(WIDTH));
		setParameter("display", "height", Integer.toString(HEIGHT));
		setParameter("display", "low resolution stride", Integer.toString(LOW_RES_STRIDE));
		setParameter("display", "unselected color", "#202080");
		setParameter("display", "selected color", "#30A030");
	}
}
