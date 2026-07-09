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

		addItemToSet("generators", "client.gui.SliderTempMutateLinksGenerator", 10);
		//addItemToSet("generators", "client.evolution.generators.AddAcyclicLink", 6);
		addItemToSet("generators", "client.evolution.generators.CrazyNewLinkAdder", 6);
		addItemToSet("generators", "client.evolution.generators.AddNodes", 4);
		//addItemToSet("generators", "client.evolution.generators.CrossoverCombiner", 3);
		addItemToSet("generators", "client.evolution.generators.MutateActivation", 1);
		
		// model the delphiNEAT for now
		setParameter("activation", "bias", "1.0");
		setParameter("activation", "x scale", Double.toString(1.0 * WIDTH / HEIGHT)); // for aspect ratio correction
		setParameter("activation", "y scale", "1.0");
		setParameter("activation", "distance scale", Double.toString(Math.sqrt(2.0)));
		
		setParameter("evolution", "weight mutation rate", "0.20");
		setParameter("evolution", "weight mutation power", "1.0");
		setParameter("evolution", "min weight mutation power", "0.01");
		setParameter("evolution", "max weight mutation power", "2.0");
		setParameter("evolution", "activation mutation rate", "0.025");
		setParameter("evolution", "max link weight", "3.0");
		setParameter("evolution", "add node rate", "0.07");
		setParameter("evolution", "add link rate", "0.10");
		setParameter("evolution", "hidden nodes", "1");
		setParameter("evolution", "hidden color nodes", "2");
		setParameter("evolution", "population size", Integer.toString(ROWS * COLUMNS));
		setParameter("evolution", "add color link rate", "0.50");
		setParameter("evolution", "add grey link rate", "0.50");
		setParameter("evolution", "add grey-color link rate", "0.10");
		
		setParameter("display", "rows", Integer.toString(ROWS));
		setParameter("display", "columns", Integer.toString(COLUMNS));
		setParameter("display", "width", Integer.toString(WIDTH));
		setParameter("display", "height", Integer.toString(HEIGHT));
		setParameter("display", "low resolution stride", Integer.toString(LOW_RES_STRIDE));
		setParameter("display", "unselected color", "#202080");
		setParameter("display", "selected color", "#30A030");
	}
}
