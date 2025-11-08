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

import client.evolution.*;

import java.util.*;
import client.Transferable;

public class StorageDatabase implements Database {
	private static final int GENERATIONS_PER_FILE = 10;
	private static final int MAX_ATTEMPTS = 5;
	private Set <String> removedNames, addedNames, serverNames;
	private Map <String, Storage> data;
	private Map <String, Exception> errors;
	
	public StorageDatabase() {
		data = new TreeMap <String, Storage> ();
		removedNames = new TreeSet <String> ();
		addedNames = new TreeSet <String> ();
		serverNames = new TreeSet <String> ();
		errors = new TreeMap <String, Exception> ();
	}
	
	private String createName(int id) {
		return Integer.toString(id / GENERATIONS_PER_FILE);
	}

	public String addGeneration(Generation g) {
		String name = createName(g.getNumber());
		
		addedNames.add(name);
		
		if(serverNames.contains(name))
			removedNames.add(name);
		
		if(!data.containsKey(name))
			data.put(name, new DefaultStorage(name));
		
		data.get(name).addGeneration(g);
		return name;
	}
	
	public void removeGeneration(Generation g) {
		String name = createName(g.getNumber());
		Storage s = data.get(name);
		s.removeGeneration(g);
		
		if(serverNames.contains(name))
			removedNames.add(name);
		
		if(s.isEmpty()) {
			data.put(name, null); // set to null to help the garbage collector
			data.remove(name);
			addedNames.remove(name);
		}
		else
			addedNames.add(name);
	}
	
	public void touchGeneration(Generation g) {
		String name = createName(g.getNumber());

		addedNames.add(name);
		if(serverNames.contains(name))
			removedNames.add(name);
	}
	
	public void ensureServerContains(String name) {
		serverNames.add(name);
	}
	
	public void clearServer() {
		serverNames.clear();
		addedNames.clear();
		removedNames.clear();
	}
	
	public synchronized Generation retrieveGeneration(String name, int number) throws client.server.FatalException, client.server.TimeoutException {
		if(!data.containsKey(name))
			prefetchGeneration(name, number);

		while(data.get(name) == null) {
			if(errors.containsKey(name)) {
				Exception e = errors.get(name);
				errors.remove(name);
				data.remove(name);
				
				if(e instanceof client.server.TimeoutException)
					throw (client.server.TimeoutException) e;
				else if(e instanceof client.server.FatalException)
					throw (client.server.FatalException) e;
				else
					throw new client.server.FatalException(e.getMessage());
			}

			try {
				wait();
			}
			catch(Exception e) {
				throw new client.server.FatalException("Error retrieving data from the server.");
			}
		}

		return data.get(name).getGeneration(number);
	}
	
	public synchronized void prefetchGeneration(final String name, final int number) {
		if(data.containsKey(name))
			return;

		data.put(name, null);
		
		final Object sync = this;
			
		new Thread(new Runnable() {
			public void run() {
				Storage s = new DefaultStorage(name);
				
				int counter = 0;
				while(counter < MAX_ATTEMPTS) {
					//System.out.println(name + " " + counter);
					try {
						java.io.InputStream xml = client.server.ServerConnectionInstance.get().loadStorage(name);
						client.utilities.XML.load(s, xml);
						xml.close();
						break;
					}
					catch(Exception e) {
						counter++;
						if(counter == MAX_ATTEMPTS)
							errors.put(name, e);
					}
				}
				
				synchronized(sync) {
					if(counter < MAX_ATTEMPTS)
						data.put(name, s);
					sync.notify();
				}
			}
		}).start();
	}
	
	public Set <String> getRemovalList() {
		return removedNames;
	}
	
	public Map <String, Transferable> getAdditionList() {
		TreeMap <String, Transferable> copy = new TreeMap <String, Transferable> ();
		
		for(String name : addedNames)
			copy.put(name, data.get(name));
		
		return copy;
	}
}
