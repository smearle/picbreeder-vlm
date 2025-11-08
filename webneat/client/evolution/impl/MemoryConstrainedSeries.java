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

package client.evolution.impl;

import client.ParameterTableInstance;
import client.evolution.EvolutionException;
import client.evolution.Individual;
import client.evolution.Genome;

/**
 * This Series implementation creates a sliding window around the current
 * position of the series.  It uses an optimal policy to remove images and
 * CPPNs from memory when the user has evolved for a long period of time.
 * <p>
 * On my machine using default jvm settings, the program computes a window
 * size of 303.  I'm assuming anoyone with more than 128 MB memory will have
 * a pretty nice window size as well.
 * 
 * @author Nick
 *
 */

class MemoryConstrainedSeries extends DefaultSeries {
	/**
	 * The amount of memory that may be taken up by images.
	 */
	private static final long MEMORY_LIMIT; // in MBs
	private static final int MEMORY_PERCENT = 5;
	
	static {
		// only use a certain percent of the max memory so our other classes may work
		long limit = Runtime.getRuntime().maxMemory() * MEMORY_PERCENT / 100;
		if(limit == Long.MAX_VALUE) // undefined
			limit = Runtime.getRuntime().freeMemory() * MEMORY_PERCENT / 100;
		MEMORY_LIMIT = limit;
	}
	
	private final int windowSize;
	
	/**
	 * The starting index of the sliding window.
	 */
	private int windowStart;
	
	public MemoryConstrainedSeries(boolean spawnPopulation) {
		super(spawnPopulation);

		windowStart = 0;
		windowSize = calculateWindowSize();
	}
	
	public MemoryConstrainedSeries(Genome branchFrom) {
		super(branchFrom);
		
		windowStart = 0;
		windowSize = calculateWindowSize();
	}
	
	private static int calculateWindowSize() {
		final int WIDTH = ParameterTableInstance.get().getInteger("display", "width");
		final int HEIGHT = ParameterTableInstance.get().getInteger("display", "height");
		final int ROWS = ParameterTableInstance.get().getInteger("display", "rows");
		final int COLUMNS = ParameterTableInstance.get().getInteger("display", "columns");
		
		long req = WIDTH * HEIGHT;
		req *= ROWS * COLUMNS;
		
		 // 2 is min because redo does back/spawn... don't want to make people wait
		return Math.max(2, (int) (MEMORY_LIMIT / req));
	}
	
	public void goForward() {
		super.goForward();
		ensureMemory();
	}
	
	public void goBack() {
		super.goBack();
		ensureMemory();
	}
	
	public void spawn() throws EvolutionException {
		super.spawn();
		ensureMemory();
	}
	
	protected void ensureMemory() {
		updateWindow();
		
		try {
			// optimal removal policy
			// pick the object in the window that is farthest away from the position.
			if(windowStart > 0)
				for(Individual ind : getGeneration(windowStart - 1))
					ind.conserveMemory();
			else if(windowStart + windowSize < getLength())
				for(Individual ind : getGeneration(windowStart + windowSize))
					ind.conserveMemory();
		}
		catch(client.server.ServerException e) {
			// should not occur
			e.printStackTrace();
		}
	}
	
	private void updateWindow() {
		final int position = getPosition();
		
		if(windowStart > position)
			windowStart = position;
		else if(windowStart + windowSize <= position)
			windowStart = position - windowSize + 1;
	}
}
